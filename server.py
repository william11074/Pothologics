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
SUPABASE_LAST_ERROR = None  # stores last insert error message (if any)

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
def supabase_insert_detection(record: dict) -> bool:
    """Insert a detection row into Supabase. Returns True on success.
    Strategy:
      1) Try as-is.
      2) If unknown-column errors mention confidence_peak, retry after mapping
         confidence_peak -> confidence and removing confidence_peak.
    On error, stores message in SUPABASE_LAST_ERROR for diagnostics."""
    global SUPABASE_LAST_ERROR
    if not supabase_enabled():
        return False
    try:
        SUPABASE.table(SUPABASE_TABLE).insert(record).execute()  # type: ignore
        SUPABASE_LAST_ERROR = None
        return True
    except Exception as e1:
        # Fallback: map confidence_peak -> confidence and REMOVE confidence_peak
        try:
            rec2 = dict(record)
            if 'confidence_peak' in rec2:
                if 'confidence' not in rec2:
                    rec2['confidence'] = rec2.get('confidence_peak')
                # Ensure we do not send unknown column
                rec2.pop('confidence_peak', None)
            SUPABASE.table(SUPABASE_TABLE).insert(rec2).execute()  # type: ignore
            SUPABASE_LAST_ERROR = None
            return True
        except Exception as e2:
            SUPABASE_LAST_ERROR = f"insert failed: {getattr(e1,'message',e1)}; fallback failed: {getattr(e2,'message',e2)}"
            try:
                print('[supabase] insert error:', SUPABASE_LAST_ERROR)
            except Exception:
                pass
            return False

# fetch all detections from Supabase table
def supabase_fetch_all_detections() -> list[dict]:
    if not supabase_enabled():
        return []
    try:
        res = SUPABASE.table(SUPABASE_TABLE).select("*").execute()
        data = getattr(res, 'data', None)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

# export Supabase detections into local csv
def supabase_export_detections_csv(path: str) -> bool:
    try:
        rows = supabase_fetch_all_detections()
        if not rows:
            return False
        cols = [
            "detection_id","device_id","cls","start_timestamp","end_timestamp","duration_s",
            "lat","lon","confidence_peak","image_url"
        ]
        out = Path(path)
        with out.open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([
                    r.get('detection_id'), r.get('device_id'), r.get('cls'), r.get('start_timestamp'),
                    r.get('end_timestamp'), r.get('duration_s'), r.get('lat'), r.get('lon'),
                    r.get('confidence_peak') or r.get('confidence'), r.get('image_url')
                ])
        return True
    except Exception:
        return False

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
            # Ok ik utcnow() is outdated but ive been using it since last year so I'm going to leave it for now
            yield f"data: {json.dumps({'type':'hello','ts':datetime.utcnow().isoformat()+'Z'})}\n\n"
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
        if 'type' not in payload:
            payload['type'] = 'routesUpdated'
        broadcast_routes_update(payload)
        return add_cors(jsonify({ 'ok': True }))
    except Exception as e:
        return add_cors(jsonify({ 'error': str(e) })), 400

# Memory tracker for detections (one row per finalized detection)
IOU_THRESHOLD = float(os.getenv("TRACK_IOU", 0.5))        # Not used when using Ultralytics IDs, kept for compatibility
MAX_MISSED_FRAMES = int(os.getenv("TRACK_MAX_MISSED", 2)) # Finalize if not seen for this many consecutive frames + 1
# Time-based finalizer: if a track hasn't updated for this many seconds, finalize it
FINALIZE_INACTIVE_SECS = float(os.getenv("TRACK_FINALIZE_SECS", 1.5))
TRACKS: dict[str, list[dict]] = {}
TRACKS_LOCK = threading.Lock()

# compute intersection of two boxes
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

# finalize track for detection
# uploads images
def _finalize_track(device_id: str, t: dict):
    start_dt = t["start_dt"]
    end_dt = t["last_dt"]
    duration_s = max(0.0, (end_dt - start_dt).total_seconds())
    start_ts = start_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    end_ts = end_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    image_url = None
    if _active_source_is_real() and t.get("best_img_bytes"):
        img_uuid = t.get("id") or str(uuid.uuid4())
        day = end_dt.strftime("%Y-%m-%d")
        img_path = f"{device_id}/{day}/{img_uuid}.jpg"
        image_url = supabase_upload_jpeg(t["best_img_bytes"], img_path)

    # Append a single track row to local CSV 
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

    # put data in Supabase only if using real data
    if _active_source_is_real():
        supabase_insert_detection({
            "detection_id": t.get("id"),
            "device_id": device_id,
            "cls": t.get("cls"),
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration_s": duration_s,
            "lat": t.get("best_lat"),
            "lon": t.get("best_lon"),
            "confidence_peak": t.get("best_conf"),
            "image_url": image_url,
        })

    try:
        schedule_recluster()
    except Exception:
        pass

# Background sweeper to finalize stale tracks even if no follow-up frames arrive
def _track_sweeper_loop():
    import time as _time
    while True:
        _time.sleep(0.5)
        now_dt = datetime.utcnow()
        to_finalize = []
        with TRACKS_LOCK:
            for device_id, dev_tracks in list(TRACKS.items()):
                still = []
                for t in dev_tracks:
                    try:
                        last_dt = t.get("last_dt")
                        if not last_dt:
                            still.append(t)
                            continue
                        idle_s = (now_dt - last_dt).total_seconds()
                        if idle_s > FINALIZE_INACTIVE_SECS:
                            to_finalize.append((device_id, t))
                        else:
                            still.append(t)
                    except Exception:
                        still.append(t)
                TRACKS[device_id] = still
        for device_id, t in to_finalize:
            try:
                _finalize_track(device_id, t)
            except Exception:
                pass

# Start sweeper thread once at import time
try:
    threading.Thread(target=_track_sweeper_loop, daemon=True).start()
except Exception:
    pass

# health check endpoint for app
# show whether Supabase is connected
@app.route("/api/supabase/status", methods=["GET"])
def supabase_status():
    info = {
        "configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "enabled": supabase_enabled(),
        "url": bool(SUPABASE_URL),
        "bucket": SUPABASE_BUCKET,
        "table": SUPABASE_TABLE,
        "tests": {},
        "last_insert_error": SUPABASE_LAST_ERROR,
    }
    if not supabase_enabled():
        return add_cors(jsonify(info))

    # storage test (upload then delete)
    try:
        try:
            SUPABASE.storage.create_bucket(SUPABASE_BUCKET, public=True) 
        except Exception:
            pass
        test_path = f"_health/{uuid.uuid4()}.txt"
        SUPABASE.storage.from_(SUPABASE_BUCKET).upload(
            test_path,
            b"ok",
            {"contentType": "text/plain", "upsert": "true"}  
        )

        try:
            SUPABASE.storage.from_(SUPABASE_BUCKET).remove(test_path) 
        except Exception:
            SUPABASE.storage.from_(SUPABASE_BUCKET).remove([test_path])
        info["tests"]["storage"] = True
    except Exception as e:
        info["tests"]["storage"] = str(e)

    try:
        SUPABASE.table(SUPABASE_TABLE).select("*").limit(1).execute()
        info["tests"]["table_select"] = True
    except Exception as e:
        info["tests"]["table_select"] = str(e)

    return add_cors(jsonify(info))

# reclustering pipeline
_recluster_lock = threading.Lock()
_recluster_pending_keys: set[str] = set()

# import class from cluster.py
def _import_optimizer():
    try:
        from Data_analytics.cluster import PotholeRepairOptimizer
        return PotholeRepairOptimizer
    except Exception:
        try:
            import sys as _sys
            p = Path("Data_analytics").resolve()
            if str(p) not in _sys.path:
                _sys.path.insert(0, str(p))
            from cluster import PotholeRepairOptimizer 
            return PotholeRepairOptimizer
        except Exception:
            return None

# run recluster process
def _run_recluster(input_csv: str = "detections.csv") -> bool:
    Optimizer = _import_optimizer()
    if Optimizer is None:
        return False
    try:
        if Path(input_csv).name == 'detections.csv' and supabase_enabled():
            supabase_export_detections_csv('detections.csv')
        opt = Optimizer(input_csv)
        opt.cluster_potholes()
        opt.optimize_routes()
        opt.generate_work_orders_csv('work_orders.csv')
        opt.generate_route_summary_csv('route_summary.csv')
        # Write meta about the source used for current work orders
        try:
            meta = {
                'source_csv': str(Path(input_csv).name if Path(input_csv).exists() else input_csv),
                'source_path': str(Path(input_csv)),
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            }
            with Path('work_orders_meta.json').open('w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # Notify clients via SSE
        broadcast_routes_update({
            'type': 'routesUpdated',
            'url': '/work_orders.csv',
            'ts': datetime.utcnow().isoformat() + 'Z'
        })
        return True
    except Exception:
        return False

def schedule_recluster_for(csv_path: str, delay_s: float = 1.0):
    """Debounce recluster by CSV path so rapid inserts coalesce into one run per source."""
    key = str(Path(csv_path))
    with _recluster_lock:
        if key in _recluster_pending_keys:
            return
        _recluster_pending_keys.add(key)

    def worker():
        import time as _time
        try:
            _time.sleep(max(0.1, float(delay_s)))
            _run_recluster(key)
        finally:
            with _recluster_lock:
                _recluster_pending_keys.discard(key)

    threading.Thread(target=worker, daemon=True).start()

def schedule_recluster(delay_s: float = 1.0):
    """Compatibility wrapper: schedule recluster for detections.csv."""
    schedule_recluster_for("detections.csv", delay_s)

@app.route('/api/recluster', methods=['POST','OPTIONS'])
def api_recluster():
    if request.method == 'OPTIONS':
        return add_cors(make_response("", 204))
    try:
        csv_in = (request.get_json(silent=True) or {}).get('csv') or 'detections.csv'
        schedule_recluster_for(csv_in, delay_s=0.1)
        return add_cors(jsonify({ 'scheduled': True }))
    except Exception as e:
        return add_cors(jsonify({ 'error': str(e) })), 400

# =========================
# Detections source toggle (real vs simulated)
# =========================
DETECTIONS_SOURCE_FILE = Path('detections_source.json')

def _load_detections_source() -> str:
    try:
        if DETECTIONS_SOURCE_FILE.exists():
            with DETECTIONS_SOURCE_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
                v = str((data or {}).get('source') or '').lower()
                if v in ('real','simulated'):
                    return v
    except Exception:
        pass
    # default
    return str(os.getenv('DETECTIONS_SOURCE', 'simulated')).lower()

def _save_detections_source(source: str) -> None:
    try:
        with DETECTIONS_SOURCE_FILE.open('w', encoding='utf-8') as f:
            json.dump({'source': source}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

@app.route('/api/detections/source', methods=['GET','POST','OPTIONS'])
def api_detections_source():
    if request.method == 'OPTIONS':
        return add_cors(make_response("", 204))
    if request.method == 'GET':
        return add_cors(jsonify({ 'source': _load_detections_source() }))
    try:
        payload = request.get_json(force=True) or {}
        src = str(payload.get('source') or '').lower()
        if src not in ('real','simulated'):
            return add_cors(jsonify({ 'error': 'source must be "real" or "simulated"' })), 400
        _save_detections_source(src)
        # Trigger a background rebuild of work orders from the selected source
        csv_path = 'detections.csv' if src == 'real' else 'detections_simulated.csv'
        schedule_recluster_for(csv_path, delay_s=0.1)
        return add_cors(jsonify({ 'ok': True, 'source': src, 'recluster_scheduled': True, 'csv': csv_path }))
    except Exception as e:
        return add_cors(jsonify({ 'error': str(e) })), 400

@app.route('/api/routes/meta', methods=['GET'])
def api_routes_meta():
    """Return metadata about the current work_orders (e.g., source CSV)."""
    meta_path = Path('work_orders_meta.json')
    if meta_path.exists():
        try:
            with meta_path.open('r', encoding='utf-8') as f:
                return add_cors(jsonify(json.load(f)))
        except Exception:
            pass
    # Fallback: unknown
    return add_cors(jsonify({ 'source_csv': None, 'source_path': None, 'updated_at': None }))

# =========================
# Mark route as fixed (remove its potholes and rebuild)
# =========================
def _haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def _remove_points_from_csv(csv_path: Path, points: list[tuple[float,float]], tol_m: float) -> int:
    if not csv_path.exists() or not points:
        return 0
    tol_km = max(0.0, float(tol_m)) / 1000.0
    removed = 0
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        # locate lat/lon columns
        cols = {h.lower(): i for i, h in enumerate(headers)}
        lat_col = cols.get('lat') or cols.get('latitude')
        lon_col = cols.get('lon') or cols.get('longitude')
        if lat_col is None or lon_col is None:
            return 0
        rows = []
        for r in reader:
            try:
                lat = float(r[lat_col]); lon = float(r[lon_col])
            except Exception:
                rows.append(r)
                continue
            close = False
            for (plat, plon) in points:
                try:
                    if _haversine_km(lat, lon, plat, plon) <= tol_km:
                        close = True
                        break
                except Exception:
                    continue
            if close:
                removed += 1
            else:
                rows.append(r)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    tmp.replace(csv_path)
    return removed

def supabase_delete_near_points(points: list[tuple[float,float]], tol_m: float) -> int:
    """Delete Supabase detections within tolerance meters of any point. Returns count deleted (best-effort)."""
    if not supabase_enabled() or not points:
        return 0
    try:
        rows = supabase_fetch_all_detections()
        if not rows:
            return 0
        tol_km = max(0.0, float(tol_m)) / 1000.0
        to_delete: list[str] = []
        for r in rows:
            try:
                lat = float(r.get('lat'))
                lon = float(r.get('lon'))
            except Exception:
                continue
            for plat, plon in points:
                try:
                    if _haversine_km(lat, lon, plat, plon) <= tol_km:
                        det_id = r.get('detection_id')
                        if det_id:
                            to_delete.append(det_id)
                        break
                except Exception:
                    continue
        deleted = 0
        if to_delete:
            try:
                SUPABASE.table(SUPABASE_TABLE).delete().in_('detection_id', to_delete).execute()  # type: ignore
                deleted = len(to_delete)
            except Exception:
                # Fallback: delete individually
                for d in to_delete:
                    try:
                        SUPABASE.table(SUPABASE_TABLE).delete().eq('detection_id', d).execute()  # type: ignore
                        deleted += 1
                    except Exception:
                        pass
        return deleted
    except Exception:
        return 0

def _remove_ids_from_csv(csv_path: Path, ids: list[str]) -> int:
    """Remove rows whose detection_id is in ids from a CSV. Returns number removed."""
    if not csv_path.exists() or not ids:
        return 0
    ids_set = set(map(str, ids))
    removed = 0
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        cols = {h: i for i, h in enumerate(headers)}
        didx = None
        # Try common header variants
        for key in ('detection_id', 'id'):
            if key in cols:
                didx = cols[key]
                break
        if didx is None:
            return 0
        rows = []
        for r in reader:
            if len(r) <= didx:
                rows.append(r)
                continue
            if str(r[didx]) in ids_set:
                removed += 1
            else:
                rows.append(r)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    tmp.replace(csv_path)
    return removed

def supabase_delete_ids(ids: list[str]) -> int:
    """Delete detections in Supabase by detection_id. Returns number requested (best-effort)."""
    if not supabase_enabled() or not ids:
        return 0
    try:
        SUPABASE.table(SUPABASE_TABLE).delete().in_('detection_id', list(map(str, ids))).execute()  # type: ignore
        return len(ids)
    except Exception:
        # Fallback: try individually
        deleted = 0
        for d in ids:
            try:
                SUPABASE.table(SUPABASE_TABLE).delete().eq('detection_id', str(d)).execute()  # type: ignore
                deleted += 1
            except Exception:
                pass
        return deleted

@app.route('/api/routes/fix', methods=['POST','OPTIONS'])
def api_fix_route():
    if request.method == 'OPTIONS':
        return add_cors(make_response("", 204))
    try:
        payload = request.get_json(force=True) or {}
        route_id = payload.get('route_id')
        if route_id is None:
            return add_cors(jsonify({ 'error': 'route_id required' })), 400
        route_id = str(route_id)
        tol_m = float(payload.get('tolerance_m', 5.0))  # default 5 meters
        # Default source: use latest work_orders source if present
        meta_src = None
        try:
            with Path('work_orders_meta.json').open('r', encoding='utf-8') as f:
                meta = json.load(f) or {}
                meta_src = meta.get('source_path') or meta.get('source_csv')
        except Exception:
            meta_src = None
        source = payload.get('source')
        if not source:
            source = meta_src or 'simulated'
        source = str(source).lower()
        # Choose CSV source
        if source == 'simulated':
            csv_path = Path('detections_simulated.csv')
        elif source == 'real':
            csv_path = Path('detections.csv')
        else:
            csv_path = Path(source)

        # Read work_orders to get points for this route
        wo = Path('work_orders.csv')
        if not wo.exists():
            return add_cors(jsonify({ 'error': 'work_orders.csv not found' })), 404
        points: list[tuple[float,float]] = []
        with wo.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get('route_id')) == route_id:
                    try:
                        lat = float(row.get('latitude'))
                        lon = float(row.get('longitude'))
                        points.append((lat, lon))
                    except Exception:
                        continue
        if not points:
            return add_cors(jsonify({ 'error': f'route {route_id} not found or has no points' })), 404

        removed = _remove_points_from_csv(csv_path, points, tol_m)
        supa_removed = 0
        if source == 'real':
            supa_removed = supabase_delete_near_points(points, tol_m)

        # Rebuild work orders from the chosen source CSV
        schedule_recluster_for(str(csv_path), delay_s=0.1)

        return add_cors(jsonify({ 'ok': True, 'route_id': route_id, 'removed': removed, 'supabase_removed': supa_removed, 'source': str(csv_path) }))
    except Exception as e:
        return add_cors(jsonify({ 'error': str(e) })), 400

# =========================
# Mark detections (potholes) as filled
# =========================
@app.route('/api/detections/fix', methods=['POST','OPTIONS'])
def api_fix_detections():
    if request.method == 'OPTIONS':
        return add_cors(make_response("", 204))
    try:
        payload = request.get_json(force=True) or {}
        ids = payload.get('detection_id') or payload.get('detection_ids')
        if ids is not None and not isinstance(ids, list):
            ids = [ids]
        lat = payload.get('lat'); lon = payload.get('lon')
        tol_m = float(payload.get('tolerance_m', 5.0))

        # Resolve source: explicit payload, else active toggle
        source = str(payload.get('source') or _load_detections_source()).lower()
        csv_path = Path('detections.csv') if source == 'real' else Path('detections_simulated.csv')

        removed_csv = 0
        removed_supa = 0

        if ids:
            removed_csv = _remove_ids_from_csv(csv_path, list(map(str, ids)))
            if source == 'real':
                removed_supa = supabase_delete_ids(list(map(str, ids)))
        elif lat is not None and lon is not None:
            try:
                plat = float(lat); plon = float(lon)
            except Exception:
                return add_cors(jsonify({ 'error': 'invalid lat/lon' })), 400
            removed_csv = _remove_points_from_csv(csv_path, [(plat, plon)], tol_m)
            if source == 'real':
                removed_supa = supabase_delete_near_points([(plat, plon)], tol_m)
        else:
            return add_cors(jsonify({ 'error': 'provide detection_id(s) or lat/lon' })), 400

        # Rebuild work orders from the chosen source CSV
        schedule_recluster_for(str(csv_path), delay_s=0.1)
        return add_cors(jsonify({ 'ok': True, 'source': source, 'removed': removed_csv, 'supabase_removed': removed_supa }))
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
ROUTE_COLORS_LOCK = threading.Lock()

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
        # Use a stable temp name to avoid suffix pitfalls on some platforms
        tmp = ROUTE_COLORS_PATH.with_name(ROUTE_COLORS_PATH.name + ".tmp")
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
        with ROUTE_COLORS_LOCK:
            colors = _load_route_colors()
            colors[rid_str] = color
            _save_route_colors(colors)
        return add_cors(jsonify({ 'ok': True, 'route_id': rid_str, 'color': color }))
    except Exception as e:
        return add_cors(jsonify({ 'error': str(e) })), 400

@app.route('/api/routes/colors/reset', methods=['POST','OPTIONS'])
def reset_route_colors():
    if request.method == 'OPTIONS':
        return add_cors(make_response("", 204))
    try:
        body = request.get_json(silent=True) or {}
        subset = body.get('route_ids')
        with ROUTE_COLORS_LOCK:
            if subset and isinstance(subset, list):
                colors = _load_route_colors()
                for rid in subset:
                    colors.pop(str(rid), None)
                _save_route_colors(colors)
            else:
                _save_route_colors({})
        return add_cors(jsonify({ 'ok': True }))
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
        # No per-frame streaming; detections are logged on finalization only

        # Run model with built-in tracker (ByteTrack) and persistent state
        import numpy as _np
        np_img = _np.array(img)
        results = MODEL.track(source=np_img, conf=0.5, verbose=False, persist=True, tracker='bytetrack.yaml')
        detections = []  # returned to client for overlay

        # Per-device track list
        with TRACKS_LOCK:
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

        # No streaming writes here

        # Update or create tracks strictly by ByteTrack tracker ID (Ultralytics)
        for det in frame_dets:
            # For overlay (doesn't include track_id on purpose)
            detections.append({k: det[k] for k in ["bbox","confidence","cls"]})
            tid = det.get("track_id")
            if tid is None:
                # No tracker assignment yet; skip aggregation this frame to avoid per-frame logging
                continue

            # With a tracker ID: look for existing track
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
                t["missed"] = int(t.get("missed", 0)) + 1
                if t["missed"] > MAX_MISSED_FRAMES:
                    _finalize_track(device_id, t)
                else:
                    still_active.append(t)
        with TRACKS_LOCK:
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
        payload = request.get_json(force=True) or {}
        # Standardized schema to match finalized detection rows
        detection_id = payload.get("detection_id") or str(uuid.uuid4())
        cls_name = payload.get("cls") or "pothole"
        ts = payload.get("timestamp") or datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z")
        start_ts = payload.get("start_timestamp") or ts
        end_ts = payload.get("end_timestamp") or ts
        try:
            duration_s = float(payload.get("duration_s") or 0.0)
        except Exception:
            duration_s = 0.0
        lat = payload.get("lat")
        lon = payload.get("lon")
        conf = payload.get("confidence_peak") or payload.get("confidence")
        device_id = request.headers.get("X-Device-ID") or request.args.get("device_id") or payload.get("device_id") or None

        out = Path("detections.csv"); new_file = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow([
                    "detection_id","device_id","cls","start_timestamp","end_timestamp","duration_s",
                    "lat","lon","confidence_peak","image_url"
                ])
            w.writerow([
                detection_id, device_id, cls_name, start_ts, end_ts, f"{duration_s:.3f}",
                lat, lon, conf, None
            ])

        # Mirror to Supabase only when using real data (avoid cross-contamination during simulation)
        if _active_source_is_real():
            supabase_insert_detection({
                "detection_id": detection_id,
                "device_id": device_id,
                "cls": cls_name,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "duration_s": duration_s,
                "lat": lat,
                "lon": lon,
                "confidence_peak": conf,
                "image_url": None,
            })
        return add_cors(jsonify({"logged": True, "detection_id": detection_id}))
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

# PWA assets served at root scope for wide SW/manifest scope
@app.route('/manifest.webmanifest')
def pwa_manifest():
    resp = send_from_directory(directory='static', path='manifest.webmanifest', as_attachment=False)
    try:
        resp.headers['Content-Type'] = 'application/manifest+json'
    except Exception:
        pass
    return add_cors(resp)

@app.route('/sw.js')
def pwa_sw():
    resp = send_from_directory(directory='static', path='sw.js', as_attachment=False)
    # Ensure correct MIME
    resp.headers['Content-Type'] = 'application/javascript'
    return add_cors(resp)

@app.route('/static/icons/icon-192.png')
def pwa_icon_192():
    p = Path('static/icons/icon-192.png')
    if p.exists():
        return add_cors(send_from_directory(directory='static/icons', path='icon-192.png', as_attachment=False))
    # Generate a simple placeholder icon
    try:
        import io as _io
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        img = _Image.new('RGBA', (192, 192), (15, 23, 42, 255))
        d = _ImageDraw.Draw(img)
        d.ellipse((24, 24, 168, 168), fill=(59,130,246,255))
        buf = _io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
        return add_cors(send_file(buf, mimetype='image/png'))
    except Exception:
        return add_cors(jsonify({ 'error': 'icon not available' })), 404

@app.route('/static/icons/icon-512.png')
def pwa_icon_512():
    p = Path('static/icons/icon-512.png')
    if p.exists():
        return add_cors(send_from_directory(directory='static/icons', path='icon-512.png', as_attachment=False))
    # Generate a simple placeholder icon
    try:
        import io as _io
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        img = _Image.new('RGBA', (512, 512), (15, 23, 42, 255))
        d = _ImageDraw.Draw(img)
        d.ellipse((64, 64, 448, 448), fill=(59,130,246,255))
        buf = _io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
        return add_cors(send_file(buf, mimetype='image/png'))
    except Exception:
        return add_cors(jsonify({ 'error': 'icon not available' })), 404

@app.route("/detections_simulated.csv", methods=["GET"])
def serve_detections_simulated_csv():
    """Serve detections_simulated.csv from the project root if it exists."""
    out = Path("detections_simulated.csv")
    if not out.exists():
        return add_cors(jsonify({"error": "detections_simulated.csv not found"})), 404
    return add_cors(send_from_directory(directory=str(out.parent), path=out.name, as_attachment=False))

@app.route("/route_summary.csv", methods=["GET"])
def serve_route_summary_csv():
    out = Path("route_summary.csv")
    if not out.exists():
        return add_cors(jsonify({"error": "route_summary.csv not found"})), 404
    return add_cors(send_from_directory(directory=str(out.parent), path=out.name, as_attachment=False))

@app.route("/detections.csv", methods=["GET"])
def serve_active_detections_csv():
    """Serve the active detections CSV based on the configured source (real or simulated)."""
    src = _load_detections_source()
    # For real data and Supabase, stream live export to avoid stale local CSV
    if src == 'real' and supabase_enabled():
        try:
            rows = supabase_fetch_all_detections()
            if not rows:
                return add_cors(jsonify({"error": "no data", "source": src})), 404
            mem = io.StringIO()
            w = csv.writer(mem)
            w.writerow(["detection_id","device_id","cls","start_timestamp","end_timestamp","duration_s","lat","lon","confidence_peak","image_url"])
            for r in rows:
                w.writerow([
                    r.get('detection_id'), r.get('device_id'), r.get('cls'), r.get('start_timestamp'),
                    r.get('end_timestamp'), r.get('duration_s'), r.get('lat'), r.get('lon'),
                    r.get('confidence_peak') or r.get('confidence'), r.get('image_url')
                ])
            mem.seek(0)
            return add_cors(send_file(io.BytesIO(mem.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=False, download_name='detections.csv'))
        except Exception as e:
            return add_cors(jsonify({"error": str(e)})), 500
    # Otherwise serve local file
    out = Path('detections.csv') if src == 'real' else Path('detections_simulated.csv')
    if not out.exists():
        return add_cors(jsonify({"error": f"{out.name} not found", "source": src})), 404
    return add_cors(send_from_directory(directory=str(out.parent), path=out.name, as_attachment=False))

@app.route("/api/detections", methods=["GET"])
def list_detections():
    """
    Return the last N rows of the local CSV (debugging convenience).
    Useful if Supabase is down or for quick inspection.
    """
    out = Path("detections.csv")
    if not out.exists():
        return add_cors(jsonify({"rows": [], "headers": ["timestamp", "lat", "lon", "confidence_peak", "device_id", "image_url"]}))
    limit = request.args.get('limit', default=200, type=int)
    rows = []
    with out.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, ["timestamp","lat","lon","confidence_peak","device_id","image_url"])
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
        writer.writerow(["timestamp","lat","lon","confidence_peak","device_id","image_url"])
        mem.seek(0)
        return add_cors(send_file(io.BytesIO(mem.getvalue().encode('utf-8')), as_attachment=True, download_name='detections.csv', mimetype='text/csv'))
    return add_cors(send_from_directory(directory=str(out.parent), path=out.name, as_attachment=True))

def _active_source_is_real() -> bool:
    try:
        return _load_detections_source() == 'real'
    except Exception:
        return False

# =========================
# Dev server
# =========================
if __name__ == "__main__":
    # In production use a proper WSGI/ASGI server (gunicorn/uvicorn)
    app.run(host="0.0.0.0", port=5000, debug=True)
