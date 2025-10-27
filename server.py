import io
import os
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, make_response, send_file, render_template
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    load_dotenv = None

# --- YOLO model ---
MODEL = "best.pt"
YOLO_OK = False
try:
    from ultralytics import YOLO
    model_path = Path("best.pt")
    if model_path.exists():
        MODEL = YOLO(str(model_path))
        YOLO_OK = True
except Exception:
    YOLO_OK = False

# --- Flask app ---
# Serve Jinja templates from ./templates and static assets from the same folder at /static
app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB frame limit

# --- Supabase config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "detections")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "detections")
SUPABASE = None

try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client  # type: ignore
        SUPABASE = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    SUPABASE = None

def supabase_enabled():
    return SUPABASE is not None

def supabase_upload_jpeg(content: bytes, path: str) -> str | None:
    try:
        if not supabase_enabled():
            return None
        try:
            SUPABASE.storage.create_bucket(SUPABASE_BUCKET, public=True)  # type: ignore
        except Exception:
            pass
        SUPABASE.storage.from_(SUPABASE_BUCKET).upload(path,
    content,
    {"contentType": "image/jpeg", "upsert": "true"})  # type: ignore
        return SUPABASE.storage.from_(SUPABASE_BUCKET).get_public_url(path)  # type: ignore
    except Exception:
        return None

def supabase_insert_detection(record: dict) -> None:
    try:
        if not supabase_enabled():
            return
        SUPABASE.table(SUPABASE_TABLE).insert(record).execute()  # type: ignore
    except Exception:
        pass

# --- Simple in-memory tracker (per device) ---
IOU_THRESHOLD = float(os.getenv("TRACK_IOU", 0.5))
MAX_MISSED_FRAMES = int(os.getenv("TRACK_MAX_MISSED", 2))
TRACKS: dict[str, list[dict]] = {}

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0

def _finalize_track(device_id: str, t: dict):
    start_dt = t["start_dt"]
    end_dt = t["last_dt"]
    duration_s = max(0.0, (end_dt - start_dt).total_seconds())
    start_ts = start_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    end_ts = end_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    # Upload best image if any
    image_url = None
    if t.get("best_img_bytes"):
        img_uuid = t.get("id") or str(uuid.uuid4())
        day = end_dt.strftime("%Y-%m-%d")
        img_path = f"{device_id}/{day}/{img_uuid}.jpg"
        image_url = supabase_upload_jpeg(t["best_img_bytes"], img_path)

    # Append to CSV as an event (single row per track)
    out = Path("detections.csv")
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["detection_id","device_id","cls","start_timestamp","end_timestamp","duration_s","lat","lon","accuracy","confidence_peak","image_url"])
        w.writerow([
            t.get("id"), device_id, t.get("cls"), start_ts, end_ts, f"{duration_s:.3f}",
            t.get("best_lat"), t.get("best_lon"), t.get("best_acc"), t.get("best_conf"), image_url
        ])

    # Mirror to Supabase table (best-effort)
    supabase_insert_detection({
        "detection_id": t.get("id"),
        "device_id": device_id,
        "cls": t.get("cls"),
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "duration_s": duration_s,
        "lat": t.get("best_lat"),
        "lon": t.get("best_lon"),
        "accuracy": t.get("best_acc"),
        "confidence": t.get("best_conf"),
        "image_url": image_url,
    })

@app.route("/api/supabase/status", methods=["GET"])
def supabase_status():
    info = {
        "configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "enabled": supabase_enabled(),
        "url": bool(SUPABASE_URL),
        "bucket": SUPABASE_BUCKET,
        "table": SUPABASE_TABLE,
        "tests": {}
    }
    if not supabase_enabled():
        return add_cors(jsonify(info))

    # Storage test: try create bucket (no-op if exists), upload and remove a tiny file
    try:
        try:
            SUPABASE.storage.create_bucket(SUPABASE_BUCKET, public=True)  # type: ignore
        except Exception:
            pass
        test_path = f"_health/{uuid.uuid4()}.txt"
        SUPABASE.storage.from_(SUPABASE_BUCKET).upload(test_path,
    b"ok",
    {"contentType": "text/plain", "upsert": "true"})  # type: ignore
        # Some client versions require list for remove
        try:
            SUPABASE.storage.from_(SUPABASE_BUCKET).remove(test_path)  # type: ignore
        except Exception:
            SUPABASE.storage.from_(SUPABASE_BUCKET).remove([test_path])  # type: ignore
        info["tests"]["storage"] = True
    except Exception as e:
        info["tests"]["storage"] = str(e)

    # Table test: simple select with limit 1
    try:
        SUPABASE.table(SUPABASE_TABLE).select("*").limit(1).execute()  # type: ignore
        info["tests"]["table_select"] = True
    except Exception as e:
        info["tests"]["table_select"] = str(e)

    return add_cors(jsonify(info))

# --- CORS ---
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Device-ID"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@app.after_request
def apply_cors(resp):
    return add_cors(resp)

# --- Routes ---
@app.route("/ping")
@app.route("/api/ping")
def ping():
    info = {"ok": True, "yolo": YOLO_OK}
    try:
        if YOLO_OK and hasattr(MODEL, 'names'):
            info["classes"] = MODEL.names
        info["model"] = str(Path("best.pt").resolve())
    except Exception:
        pass
    return add_cors(jsonify(info))

@app.route("/")
def root():
    return render_template("index.html")

@app.route("/api/detect", methods=["POST", "OPTIONS"])
def api_detect():
    if request.method == "OPTIONS":
        return add_cors(make_response("", 204))

    file = request.files.get("image")
    if not file:
        return add_cors(jsonify({"error": "no image"})), 400
    if not YOLO_OK:
        return add_cors(jsonify({"detections": [], "yolo": False}))

    try:
        data = file.read()
        import PIL.Image as Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size

        # Optional metadata
        device_id = request.form.get("device_id") or request.headers.get("X-Device-ID") or "unknown"
        def _f(name):
            v = request.form.get(name)
            try:
                return float(v) if v not in (None, "") else None
            except Exception:
                return None
        lat = _f("lat"); lon = _f("lon"); accuracy = _f("accuracy")
        client_ts = request.form.get("timestamp")
        now_dt = datetime.utcnow()

        # confidence threshold 0.5
        # Use Ultralytics built-in tracker with persistence to obtain stable track IDs
        import numpy as _np
        np_img = _np.array(img)
        results = MODEL.track(source=np_img, conf=0.5, verbose=False, persist=True, tracker='bytetrack.yaml')
        detections = []

        # Per-device tracking
        dev_tracks = TRACKS.setdefault(device_id, [])
        matched_track_ids = set()

        def crop_bytes(bbox):
            try:
                x1, y1, x2, y2 = map(int, bbox)
                c = img.crop((x1, y1, x2, y2))
                buf = io.BytesIO(); c.save(buf, format="JPEG", quality=85)
                return buf.getvalue()
            except Exception:
                return data

        # Collect detections for this frame (with Ultralytics track IDs)
        frame_dets = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes
            n = len(boxes)
            for j in range(n):
                try:
                    xyxy = boxes.xyxy[j].tolist()
                    conf = float(boxes.conf[j].item())
                    cls_id = int(boxes.cls[j].item()) if boxes.cls is not None else 0
                    track_id = int(boxes.id[j].item()) if boxes.id is not None else None
                    cls_name = (MODEL.names or {}).get(cls_id, "pothole")
                except Exception:
                    continue
                x1, y1, x2, y2 = xyxy
                x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
                frame_dets.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "cls": cls_name, "track_id": track_id})

        # Assign by Ultralytics-provided track IDs
        frame_dets.sort(key=lambda d: d["confidence"], reverse=True)
        for det in frame_dets:
            # For overlay on client
            detections.append({k: det[k] for k in ["bbox","confidence","cls"]})
            tid = det.get("track_id")
            if tid is None:
                continue
            # Find existing track by tid
            t = next((x for x in dev_tracks if x.get("track_id") == tid), None)
            if t is None:
                t = {
                    "id": str(uuid.uuid4()),  # external event id we log
                    "track_id": tid,
                    "cls": det["cls"],
                    "bbox": det["bbox"],
                    "start_dt": now_dt,
                    "last_dt": now_dt,
                    "missed": 0,
                    "best_conf": det["confidence"],
                    "best_img_bytes": crop_bytes(det["bbox"]),
                    "best_lat": lat, "best_lon": lon, "best_acc": accuracy,
                }
                dev_tracks.append(t)
            else:
                t["bbox"] = det["bbox"]
                t["last_dt"] = now_dt
                t["missed"] = 0
                if det["confidence"] >= (t.get("best_conf") or 0):
                    t["best_conf"] = det["confidence"]
                    t["best_img_bytes"] = crop_bytes(det["bbox"])
                    t["best_lat"], t["best_lon"], t["best_acc"] = lat, lon, accuracy
            matched_track_ids.add(tid)

        # Age and finalize tracks not matched this frame
        still_active = []
        for t in dev_tracks:
            if t.get("track_id") in matched_track_ids:
                still_active.append(t)
            else:
                t["missed"] += 1
                if t["missed"] > MAX_MISSED_FRAMES:
                    _finalize_track(device_id, t)
                else:
                    still_active.append(t)
        TRACKS[device_id] = still_active

        return add_cors(jsonify({"detections": detections, "size": [w, h], "yolo": True}))
    except Exception as e:
        return add_cors(jsonify({"error": str(e)})), 500

@app.route("/api/log", methods=["POST", "OPTIONS"])
def api_log():
    if request.method == "OPTIONS":
        return add_cors(make_response("", 204))
    try:
        payload = request.get_json(force=True)
        ts = payload.get("timestamp") or datetime.utcnow().isoformat()
        lat = payload.get("lat"); lon = payload.get("lon"); acc = payload.get("accuracy"); conf = payload.get("confidence")
        out = Path("detections.csv"); new_file = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "lat", "lon", "accuracy", "confidence", "device_id", "image_url"])
            device_id = request.headers.get("X-Device-ID") or request.args.get("device_id") or None
            w.writerow([ts, lat, lon, acc, conf, device_id, None])
        # Mirror to Supabase (no image)
        supabase_insert_detection({
            "timestamp": ts, "device_id": device_id, "lat": lat, "lon": lon, "accuracy": acc, "confidence": conf, "image_url": None,
        })
        return add_cors(jsonify({"logged": True}))
    except Exception as e:
        return add_cors(jsonify({"error": str(e)})), 400

@app.route("/detect.html")
@app.route("/detect")
def serve_detect():
    return render_template("detect.html")

@app.route("/routes")
@app.route("/routes.html")
def serve_routes():
    return render_template("routes.html")

@app.route("/api/detections", methods=["GET"])
def list_detections():
    out = Path("detections.csv")
    if not out.exists():
        return add_cors(jsonify({"rows": [], "headers": ["timestamp", "lat", "lon", "accuracy", "confidence", "device_id", "image_url"]}))
    limit = request.args.get('limit', default=200, type=int)
    rows = []
    with out.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, ["timestamp","lat","lon","accuracy","confidence","device_id","image_url"])
        for r in reader:
            rows.append(r)
    if limit and limit > 0:
        rows = rows[-limit:]
    return add_cors(jsonify({"headers": headers, "rows": rows}))

@app.route("/download/detections.csv", methods=["GET"])
def download_detections():
    out = Path("detections.csv")
    if not out.exists():
        mem = io.StringIO(); writer = csv.writer(mem)
        writer.writerow(["timestamp","lat","lon","accuracy","confidence","device_id","image_url"])
        mem.seek(0)
        return add_cors(send_file(io.BytesIO(mem.getvalue().encode('utf-8')), as_attachment=True, download_name='detections.csv', mimetype='text/csv'))
    return add_cors(send_from_directory(directory=str(out.parent), path=out.name, as_attachment=True))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
