# app.py
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = (BASE_DIR / ".." / "frontend").resolve()  # <-- where interviewer.html lives

# Serve static assets from ../frontend
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# -------- Static pages --------
@app.route("/")
def root():
    # default to interviewer dashboard
    return send_from_directory(FRONTEND_DIR, "interviewer.html")

@app.route("/interviewer.html")
def interviewer_page():
    return send_from_directory(FRONTEND_DIR, "interviewer.html")

@app.route("/candidate.html")
def candidate_page():
    return send_from_directory(FRONTEND_DIR, "candidate.html")

# Optional: serve any other assets in ../frontend (css/js/images, etc.)
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# -------- Simple in-memory session store --------
sessions = defaultdict(lambda: {
    "last": None,            # last payload from candidate
    "prediction_count": 0,   # how many times we've predicted
})

# -------- Health endpoint --------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "model_loaded": True,     # swap to your real model state if needed
        "model_type": "gaze_rf_v1"
    })

# -------- Candidate posts each frame --------
@app.route("/api/update_gaze", methods=["POST"])
def update_gaze():
    data = request.get_json(force=True)
    sid = data.get("sessionId")  # Fixed: handle both sessionId and session_id
    if not sid:
        sid = data.get("session_id")  # fallback
    if not sid:
        return jsonify({"error": "missing sessionId"}), 400
    
    print(f"Received data for session {sid}: {data}")  # Debug log
    
    # very light "feature quality" calc + fake prediction
    iris = data.get("irisFeatures") or []
    feature_quality = 0.0
    if isinstance(iris, list) and len(iris) == 4:
        try:
            within = [0.0 <= float(v) <= 1.0 for v in iris]
        except Exception:
            within = [False, False, False, False]
        feature_quality = sum(within) / 4.0

    # naive prediction so the UI can move (replace with your model)
    pred = "center"
    if isinstance(iris, list) and len(iris) == 4:
        try:
            lx, rx, uy, ly = [float(v) for v in iris]
        except Exception:
            lx = rx = uy = ly = 0.5
        if lx < 0.35:
            pred = "left"
        elif rx < 0.35:
            pred = "right"
        elif uy < 0.4:
            pred = "up"
        elif ly < 0.4:
            pred = "down"
        else:
            pred = "center"

    # Check if face is detected
    face_detected = bool(data.get("faceDetected", False))
    
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "face_detected": face_detected,
        "landmark_quality": float(data.get("landmarkConfidence", 0.0)),
        "feature_quality": float(feature_quality),
        "raw_prediction": pred if face_detected else "no_face",
        "detection_method": data.get("detectionMethod") or "simple_cv",
        "mediapipe_active": (data.get("detectionMethod") == "mediapipe"),
        "iris": iris
    }

    sessions[sid]["last"] = payload
    sessions[sid]["prediction_count"] += 1
    
    print(f"Stored payload for session {sid}: {payload}")  # Debug log
    
    return jsonify({"ok": True, "received": True})

# -------- Interviewer polls every second --------
@app.route("/api/gaze_status", methods=["GET"])
def gaze_status():
    sid = request.args.get("sessionId")
    if not sid or sid not in sessions or sessions[sid]["last"] is None:
        # exactly what the UI expects when nothing is available yet
        return jsonify({"raw_prediction": "no_data"})

    last = sessions[sid]["last"]
    # age in seconds
    try:
        ts = datetime.fromisoformat(last["timestamp"])
    except Exception:
        ts = datetime.now(timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds()

    out = {
        "raw_prediction": last["raw_prediction"],
        "face_detected": last["face_detected"],
        "feature_quality": last["feature_quality"],
        "landmark_quality": last["landmark_quality"],
        "detection_method": last["detection_method"],
        "mediapipe_active": last["mediapipe_active"],
        "prediction_count": sessions[sid]["prediction_count"],
        "data_age_seconds": age
    }
    return jsonify(out)

# -------- Optional: "Explain Last" mock --------
@app.route("/api/explain_last", methods=["GET"])
def explain_last():
    sid = request.args.get("sessionId")
    if not sid or sid not in sessions or sessions[sid]["last"] is None:
        return jsonify({"message": "no_data"})

    last = sessions[sid]["last"]
    iris = last.get("iris") or [0, 0, 0, 0]
    try:
        iris = [float(v) for v in iris]
    except Exception:
        iris = [0, 0, 0, 0]

    shap_vals = [
        {"feature": "left_x",  "impact": (0.5 - iris[0])},
        {"feature": "right_x", "impact": (0.5 - iris[1]) * -1},
        {"feature": "up_y",    "impact": (0.5 - iris[2])},
        {"feature": "low_y",   "impact": (0.5 - iris[3]) * -1},
    ]

    lime = {"weights": [
        {"feature": "left_x",  "weight": shap_vals[0]["impact"]},
        {"feature": "right_x", "weight": shap_vals[1]["impact"]},
        {"feature": "up_y",    "weight": shap_vals[2]["impact"]},
        {"feature": "low_y",   "weight": shap_vals[3]["impact"]},
    ]}

    proba = [0.1, 0.1, 0.1, 0.1, 0.6]  # mock probas
    return jsonify({
        "predicted_label": last["raw_prediction"],
        "proba": proba,
        "top_reasons": sorted(shap_vals, key=lambda x: abs(x["impact"]), reverse=True)[:4],
        "lime": lime
    })

# -------- Debug endpoint to see all sessions --------
@app.route("/api/debug/sessions", methods=["GET"])
def debug_sessions():
    """Debug endpoint to see all active sessions"""
    debug_data = {}
    for sid, data in sessions.items():
        debug_data[sid] = {
            "has_data": data["last"] is not None,
            "prediction_count": data["prediction_count"],
            "last_timestamp": data["last"]["timestamp"] if data["last"] else None,
            "last_prediction": data["last"]["raw_prediction"] if data["last"] else None
        }
    return jsonify(debug_data)

# -------- Main --------
if __name__ == "__main__":
    print(f"Serving frontend from: {FRONTEND_DIR}")
    print("Open: http://127.0.0.1:5000   (note: http, not https)")
    print("Debug endpoint: http://127.0.0.1:5000/api/debug/sessions")
    app.run(host="0.0.0.0", port=5000, debug=True)