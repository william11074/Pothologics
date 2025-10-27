import io
import os
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, make_response, send_file, render_template

# Load API keys from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    load_dotenv = None

# Load YOLO model
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

# Set up flask app
app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Set up Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "detections")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "detections")
SUPABASE = None

try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        SUPABASE = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    SUPABASE = None

def supabase_enabled():
    return SUPABASE is not None

# Upload JPEG to Supabase storage (pothole detection images)
def supabase_upload_jpeg(content: bytes, path: str) -> str | None:
    try:
        if not supabase_enabled():
            return None
        try:
            SUPABASE.storage.create_bucket(SUPABASE_BUCKET, public=True)
        except Exception:
            pass
        SUPABASE.storage.from_(SUPABASE_BUCKET).upload( 
            path,
            content,
            {"contentType": "image/jpeg", "upsert": "true"}
        )
        return SUPABASE.storage.from_(SUPABASE_BUCKET).get_public_url(path)
    except Exception:
        return None

# insert detection record into Supabase table
def supabase_insert_detection(record: dict) -> None:
    try:
        if not supabase_enabled():
            return
        SUPABASE.table(SUPABASE_TABLE).insert(record).execute()
    except Exception:
        pass

# Server sent events (SSE) broadcaster (for live update of the map)
import queue
import threading

_listeners_lock = threading.Lock()
_listeners: list[queue.SimpleQueue] = []

def broadcast_routes_update(payload: dict):
    with _listeners_lock:
        for q in list(_listeners):
            try:
                q.put(payload)
            except Exception:
                pass

# SSE endpoint for clients to receive live updates
@app.route('/events')
def sse_events():
    def gen():
        q: queue.SimpleQueue = queue.SimpleQueue()
        with _listeners_lock:
            _listeners.append(q)
        try:
            # Initial ping
            yield f"data: {json.dumps({'type':'hello','ts':datetime.utcnow().isoformat()+'Z'})}\n\n"
            # Heartbeat
            import time
            last_beat = time.time()
            while True:
                try:
                    item = q.get(timeout=5)
                    yield f"data: {json.dumps(item)}\n\n"
                except Exception:
                    if time.time() - last_beat > 20:
                        yield "data: {\"type\":\"heartbeat\"}\n\n"
                        last_beat = time.time()
        finally:
            with _listeners_lock:
                if q in _listeners:
                    _listeners.remove(q)

    resp = app.response_class(gen(), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

# Broadcast whenever a route is updated
@app.route('/api/routes/notify', methods=['POST'])
def notify_routes():
    try:
        payload = request.get_json(force=True) or {}
        # Assign a default type if not provided
        if 'type' not in payload:
            payload['type'] = 'routesUpdated'
        broadcast_routes_update(payload)
        return add_cors(jsonify({ 'ok': True }))
    except Exception as e:
        return add_cors(jsonify({ 'error': str(e) })), 400

# Memory tracker for detections
IOU_THRESHOLD = float(os.getenv("TRACK_IOU", 0.5))        # Not used when using Ultralytics IDs, kept for compatibility
MAX_MISSED_FRAMES = int(os.getenv("TRACK_MAX_MISSED", 2)) # Finalize if not seen for this many consecutive frames + 1
TRACKS: dict[str, list[dict]] = {}

def _iou(a, b):
    """Compute intersection-over-union (unused in current assignment by track ID)."""
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
    """
    Finalize a track:
    - compute duration
    - upload best crop to storage (if present)
    - append a single row to local CSV
    - mirror to Supabase table
    """
    start_dt = t["start_dt"]
    end_dt = t["last_dt"]
    duration_s = max(0.0, (end_dt - start_dt).total_seconds())
    # Normalize to RFC3339 UTC
    start_ts = start_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    end_ts = end_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    # Upload the best frame crop, if we captured one
    image_url = None
    if t.get("best_img_bytes"):
        img_uuid = t.get("id") or str(uuid.uuid4())
        day = end_dt.strftime("%Y-%m-%d")
        img_path = f"{device_id}/{day}/{img_uuid}.jpg"
        image_url = supabase_upload_jpeg(t["best_img_bytes"], img_path)

    # Append a single, finalized track row to local CSV (debug-friendly)
    out = Path("detections.csv")
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "detection_id","device_id","cls","start_timestamp","end_timestamp",
                "duration_s","lat","lon","confidence_peak","image_url"
            ])
        w.writerow([
            t.get("id"), device_id, t.get("cls"), start_ts, end_ts, f"{duration_s:.3f}",
            t.get("best_lat"), t.get("best_lon"), t.get("best_conf"), image_url
        ])

    # Mirror to Supabase table (single row per finalized object)
    supabase_insert_detection({
        "detection_id": t.get("id"),
        "device_id": device_id,
        "cls": t.get("cls"),
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "duration_s": duration_s,
        "lat": t.get("best_lat"),
        "lon": t.get("best_lon"),
        # NOTE: your table column is 'confidence_peak'; match that name here if needed
        "confidence": t.get("best_conf"),
        "image_url": image_url,
    })

    # Kick off a background recluster + route rebuild (debounced)
    try:
        schedule_recluster()
    except Exception:
        pass

# =========================
# Health check for Supabase
# =========================
@app.route("/api/supabase/status", methods=["GET"])
def supabase_status():
    """
    Show whether Supabase is configured and reachable.
    Also attempts a small storage upload/remove and a table select(*).
    """
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

    # Storage test: create bucket (no-op if exists), upload small file, then remove it
    try:
        try:
            SUPABASE.storage.create_bucket(SUPABASE_BUCKET, public=True)  # type: ignore
        except Exception:
            pass
        test_path = f"_health/{uuid.uuid4()}.txt"
        SUPABASE.storage.from_(SUPABASE_BUCKET).upload(  # type: ignore
            test_path,
            b"ok",
            {"contentType": "text/plain", "upsert": "true"}  # strings to satisfy some client versions
        )
        # Some client versions require list for remove
        try:
            SUPABASE.storage.from_(SUPABASE_BUCKET).remove(test_path)  # type: ignore
        except Exception:
            SUPABASE.storage.from_(SUPABASE_BUCKET).remove([test_path])  # type: ignore
        info["tests"]["storage"] = True
    except Exception as e:
        info["tests"]["storage"] = str(e)

    # Table test: basic select
    try:
        SUPABASE.table(SUPABASE_TABLE).select("*").limit(1).execute()  # type: ignore
        info["tests"]["table_select"] = True
    except Exception as e:
        info["tests"]["table_select"] = str(e)

    return add_cors(jsonify(info))

# =========================
# Background recluster pipeline
# =========================
_recluster_lock = threading.Lock()
_recluster_pending = False

def _import_optimizer():
    try:
        # Try package-style import first
        from Data_analytics.cluster import PotholeRepairOptimizer  # type: ignore
        return PotholeRepairOptimizer
    except Exception:
        try:
            import sys as _sys
            p = Path("Data_analytics").resolve()
            if str(p) not in _sys.path:
                _sys.path.insert(0, str(p))
            from cluster import PotholeRepairOptimizer  # type: ignore
            return PotholeRepairOptimizer
        except Exception:
            return None

def _run_recluster(input_csv: str = "detections.csv") -> bool:
    Optimizer = _import_optimizer()
    if Optimizer is None:
        return False
    try:
        opt = Optimizer(input_csv)
        opt.cluster_potholes()
        opt.optimize_routes()
        opt.generate_work_orders_csv('work_orders.csv')
        opt.generate_route_summary_csv('route_summary.csv')
        # Notify clients via SSE
        broadcast_routes_update({
            'type': 'routesUpdated',
            'url': '/work_orders.csv',
            'ts': datetime.utcnow().isoformat() + 'Z'
        })
        return True
    except Exception:
        return False

def schedule_recluster(delay_s: float = 1.0):
    """Debounce recluster job so rapid inserts coalesce into one run."""
    global _recluster_pending
    with _recluster_lock:
        if _recluster_pending:
            return
        _recluster_pending = True

    def worker():
        import time as _time
        try:
            _time.sleep(max(0.1, float(delay_s)))
            _run_recluster("detections.csv")
        finally:
            global _recluster_pending
            with _recluster_lock:
                _recluster_pending = False

    threading.Thread(target=worker, daemon=True).start()

@app.route('/api/recluster', methods=['POST','OPTIONS'])
def api_recluster():
    if request.method == 'OPTIONS':
        return add_cors(make_response("", 204))
    try:
        schedule_recluster(delay_s=0.1)
        return add_cors(jsonify({ 'scheduled': True }))
    except Exception as e:
        return add_cors(jsonify({ 'error': str(e) })), 400

# =========================
# CORS helpers
# =========================
def add_cors(resp):
    """Add permissive CORS headers (handy for local/mobile clients)."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Device-ID"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@app.after_request
def apply_cors(resp):
    """Apply CORS headers to every response."""
    return add_cors(resp)

# =========================
# Misc routes
# =========================
@app.route("/ping")
@app.route("/api/ping")
def ping():
    """Lightweight liveness + YOLO status."""
    info = {"ok": True, "yolo": YOLO_OK}
    try:
        if YOLO_OK and hasattr(MODEL, 'names'):
            info["classes"] = MODEL.names
        info["model"] = str(Path("best.pt").resolve())
    except Exception:
        pass
    return add_cors(jsonify(info))

# =========================
# Route color persistence
# =========================
ROUTE_COLORS_PATH = Path("route_colors.json")

def _load_route_colors() -> dict[str, str]:
    try:
        if ROUTE_COLORS_PATH.exists():
            with ROUTE_COLORS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # normalize keys to strings
                return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}

def _save_route_colors(colors: dict[str, str]) -> None:
    try:
        tmp = ROUTE_COLORS_PATH.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(colors, f, ensure_ascii=False, indent=2)
        tmp.replace(ROUTE_COLORS_PATH)
    except Exception:
        pass

def _valid_hex_color(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) != 7 or not value.startswith('#'):
        return False
    try:
        int(value[1:], 16)
        return True
    except Exception:
        return False

@app.route('/api/routes/colors', methods=['GET'])
def get_route_colors():
    colors = _load_route_colors()
    return add_cors(jsonify({ 'colors': colors }))

@app.route('/api/routes/colors', methods=['POST','OPTIONS'])
def set_route_color():
    if request.method == 'OPTIONS':
        return add_cors(make_response("", 204))
    try:
        payload = request.get_json(force=True) or {}
        rid = payload.get('route_id')
        color = payload.get('color')
        if rid is None or color is None:
            return add_cors(jsonify({ 'error': 'route_id and color required' })), 400
        rid_str = str(rid)
        if not _valid_hex_color(color):
            return add_cors(jsonify({ 'error': 'invalid color format, expected #RRGGBB' })), 400
        colors = _load_route_colors()
        colors[rid_str] = color
        _save_route_colors(colors)
        return add_cors(jsonify({ 'ok': True, 'route_id': rid_str, 'color': color }))
    except Exception as e:
        return add_cors(jsonify({ 'error': str(e) })), 400

@app.route("/")
def root():
    """Home page (renders templates/index.html)."""
    return render_template("index.html")

# =========================
# Detection endpoint (per frame)
# =========================
@app.route("/api/detect", methods=["POST", "OPTIONS"])
def api_detect():
    """
    Accept a single image frame:
    - run YOLO with ByteTrack for track IDs
    - update in-memory tracks per device
    - finalize + log a track only after it's missing for MAX_MISSED_FRAMES
    - return detections for overlay (bbox/conf/cls)
    """
    if request.method == "OPTIONS":
        return add_cors(make_response("", 204))

    file = request.files.get("image")
    if not file:
        return add_cors(jsonify({"error": "no image"})), 400
    if not YOLO_OK:
        return add_cors(jsonify({"detections": [], "yolo": False}))

    try:
        # Decode image to PIL
        data = file.read()
        import PIL.Image as Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size

        # Optional metadata from form/headers
        device_id = request.form.get("device_id") or request.headers.get("X-Device-ID") or "unknown"
        def _f(name):
            v = request.form.get(name)
            try:
                return float(v) if v not in (None, "") else None
            except Exception:
                return None
        lat = _f("lat"); lon = _f("lon")
        client_ts = request.form.get("timestamp")
        now_dt = datetime.utcnow()  # naive UTC is OK; we normalize when finalizing

        # Run model with built-in tracker (ByteTrack) and persistent state
        import numpy as _np
        np_img = _np.array(img)
        results = MODEL.track(source=np_img, conf=0.5, verbose=False, persist=True, tracker='bytetrack.yaml')
        detections = []  # returned to client for overlay

        # Per-device track list
        dev_tracks = TRACKS.setdefault(device_id, [])
        matched_track_ids = set()

        def crop_bytes(bbox):
            """Return JPEG bytes for the given XYXY bbox; fallback to original bytes on error."""
            try:
                x1, y1, x2, y2 = map(int, bbox)
                c = img.crop((x1, y1, x2, y2))
                buf = io.BytesIO(); c.save(buf, format="JPEG", quality=85)
                return buf.getvalue()
            except Exception:
                return data

        # Collect current-frame detections (including tracker IDs if available)
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
                # Clamp to image bounds
                x1, y1, x2, y2 = xyxy
                x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
                frame_dets.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "cls": cls_name, "track_id": track_id})

        # Sort by confidence so the "best" frame tends to overwrite earlier crops
        frame_dets.sort(key=lambda d: d["confidence"], reverse=True)

        # Update or create tracks by tracker ID
        for det in frame_dets:
            # For overlay (doesn't include track_id on purpose)
            detections.append({k: det[k] for k in ["bbox","confidence","cls"]})
            tid = det.get("track_id")
            if tid is None:
                # If tracker hasn't assigned an ID yet, skip track aggregation for this det
                continue

            # Look for an existing track with the same track_id
            t = next((x for x in dev_tracks if x.get("track_id") == tid), None)
            if t is None:
                # Start a new track for this object
                t = {
                    "id": str(uuid.uuid4()),  # external event id we log once per finalized track
                    "track_id": tid,
                    "cls": det["cls"],
                    "bbox": det["bbox"],
                    "start_dt": now_dt,
                    "last_dt": now_dt,
                    "missed": 0,
                    "best_conf": det["confidence"],
                    "best_img_bytes": crop_bytes(det["bbox"]),
                    "best_lat": lat, "best_lon": lon,
                }
                dev_tracks.append(t)
            else:
                # Update existing track fresh data
                t["bbox"] = det["bbox"]
                t["last_dt"] = now_dt
                t["missed"] = 0
                # Keep the best-confidence snapshot and GPS
                if det["confidence"] >= (t.get("best_conf") or 0):
                    t["best_conf"] = det["confidence"]
                    t["best_img_bytes"] = crop_bytes(det["bbox"])
                    t["best_lat"], t["best_lon"] = lat, lon
            matched_track_ids.add(tid)

        # Age tracks not matched this frame; finalize those that have been missing for too long
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

        # Return overlay info and image dimensions
        return add_cors(jsonify({"detections": detections, "size": [w, h], "yolo": True}))
    except Exception as e:
        # Return error in JSON; logs/tracebacks can be added here if desired
        return add_cors(jsonify({"error": str(e)})), 500

# =========================
# Simple JSON logger (optional)
# =========================
@app.route("/api/log", methods=["POST", "OPTIONS"])
def api_log():
    """
    Accepts a JSON payload and appends it to local CSV, mirrors to Supabase.
    Useful for quickly validating table connectivity without running YOLO.
    """
    if request.method == "OPTIONS":
        return add_cors(make_response("", 204))
    try:
        payload = request.get_json(force=True)
        ts = payload.get("timestamp") or datetime.utcnow().isoformat()
        lat = payload.get("lat"); lon = payload.get("lon"); conf = payload.get("confidence")
        out = Path("detections.csv"); new_file = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "lat", "lon", "confidence", "device_id", "image_url"])
            device_id = request.headers.get("X-Device-ID") or request.args.get("device_id") or None
            w.writerow([ts, lat, lon, conf, device_id, None])
        # Mirror to Supabase (fields may be NULL; table should allow it)
        supabase_insert_detection({
            "timestamp": ts, "device_id": device_id, "lat": lat, "lon": lon, "confidence": conf, "image_url": None,
        })
        return add_cors(jsonify({"logged": True}))
    except Exception as e:
        return add_cors(jsonify({"error": str(e)})), 400

# =========================
# Convenience routes
# =========================
@app.route("/detect.html")
@app.route("/detect")
def serve_detect():
    """Demo page for detection (templates/detect.html)."""
    return render_template("detect.html")

@app.route("/routes")
@app.route("/routes.html")
def serve_routes():
    """Demo page that can subscribe to SSE route updates (templates/routes.html)."""
    return render_template("routes.html")

@app.route("/map")
@app.route("/map.html")
def serve_map():
    """Lightweight embedded map view (templates/map.html)."""
    return render_template("map.html")

@app.route("/work_orders.csv", methods=["GET"])
def serve_work_orders_csv():
    """Serve work_orders.csv from the project root if it exists."""
    out = Path("work_orders.csv")
    if not out.exists():
        return add_cors(jsonify({"error": "work_orders.csv not found"})), 404
    return add_cors(send_from_directory(directory=str(out.parent), path=out.name, as_attachment=False))

@app.route("/detections_simulated.csv", methods=["GET"])
def serve_detections_simulated_csv():
    """Serve detections_simulated.csv from the project root if it exists."""
    out = Path("detections_simulated.csv")
    if not out.exists():
        return add_cors(jsonify({"error": "detections_simulated.csv not found"})), 404
    return add_cors(send_from_directory(directory=str(out.parent), path=out.name, as_attachment=False))

@app.route("/api/detections", methods=["GET"])
def list_detections():
    """
    Return the last N rows of the local CSV (debugging convenience).
    Useful if Supabase is down or for quick inspection.
    """
    out = Path("detections.csv")
    if not out.exists():
        return add_cors(jsonify({"rows": [], "headers": ["timestamp", "lat", "lon", "confidence", "device_id", "image_url"]}))
    limit = request.args.get('limit', default=200, type=int)
    rows = []
    with out.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, ["timestamp","lat","lon","confidence","device_id","image_url"])
        for r in reader:
            rows.append(r)
    if limit and limit > 0:
        rows = rows[-limit:]
    return add_cors(jsonify({"headers": headers, "rows": rows}))

@app.route("/download/detections.csv", methods=["GET"])
def download_detections():
    """Download the local CSV (creates a header-only CSV if none exists yet)."""
    out = Path("detections.csv")
    if not out.exists():
        mem = io.StringIO(); writer = csv.writer(mem)
        writer.writerow(["timestamp","lat","lon","confidence","device_id","image_url"])
        mem.seek(0)
        return add_cors(send_file(io.BytesIO(mem.getvalue().encode('utf-8')), as_attachment=True, download_name='detections.csv', mimetype='text/csv'))
    return add_cors(send_from_directory(directory=str(out.parent), path=out.name, as_attachment=True))

# =========================
# Dev server
# =========================
if __name__ == "__main__":
    # In production use a proper WSGI/ASGI server (gunicorn/uvicorn)
    app.run(host="0.0.0.0", port=5000, debug=True)
