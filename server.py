import io
import csv
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, make_response

MODEL = "best.pt"
MODEL_INPUT = None
YOLO_OK = False

try:
    from ultralytics import YOLO
    model_path = Path("best.pt")
    if model_path.exists():
        MODEL = YOLO(str(model_path))
        YOLO_OK = True
except Exception as e:
    YOLO_OK = False

app = Flask(__name__, static_url_path="", static_folder=".")

def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@app.after_request
def apply_cors(resp):
    return add_cors(resp)

@app.route("/ping")
@app.route("/api/ping")
def ping():
    return add_cors(jsonify({"ok": True, "yolo": YOLO_OK}))

@app.route("/")
def root():
    # Serve landing page
    return send_from_directory(".", "index.html")

@app.route("/api/detect", methods=["POST", "OPTIONS"])
def api_detect():
    if request.method == "OPTIONS":
        return add_cors(make_response("", 204))

    file = request.files.get("image")
    if not file:
        return add_cors(jsonify({"error": "no image"})), 400
    if not YOLO_OK:
        # Graceful fallback: return no detections
        return add_cors(jsonify({"detections": [], "yolo": False}))

    try:
        # Read image
        data = file.read()
        import PIL.Image as Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size

        # Inference (conf threshold can be tuned)
        results = MODEL.predict(source=img, conf=0.25, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                try:
                    xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                    conf = float(b.conf[0].item())
                    cls_id = int(b.cls[0].item()) if b.cls is not None else 0
                    cls_name = (MODEL.names or {}).get(cls_id, "pothole")
                except Exception:
                    continue
                # Clamp to image size
                x1, y1, x2, y2 = xyxy
                x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "cls": cls_name
                })
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
        lat = payload.get("lat")
        lon = payload.get("lon")
        acc = payload.get("accuracy")
        conf = payload.get("confidence")
        out = Path("detections.csv")
        new_file = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "lat", "lon", "accuracy", "confidence"])
            w.writerow([ts, lat, lon, acc, conf])
        return add_cors(jsonify({"logged": True}))
    except Exception as e:
        return add_cors(jsonify({"error": str(e)})), 400

@app.route("/detect.html")
@app.route("/detect")
def serve_detect():
    return send_from_directory(".", "detect.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
