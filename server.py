import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pymongo
from bson import ObjectId

# Initialize Flask Application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "aero3d_db"

mongo_client = None
db = None

def get_db():
    global mongo_client, db
    if db is None:
        try:
            mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            # Test connection
            mongo_client.server_info()
            db = mongo_client[DB_NAME]
            print(f"[MONGODB] Connected successfully to '{DB_NAME}' at {MONGO_URI}")
        except Exception as err:
            print(f"[MONGODB ERROR] Could not connect to MongoDB: {err}")
            return None
    return db

# Custom JSON Encoder for BSON ObjectId
def serialize_doc(doc):
    if not doc:
        return doc
    if isinstance(doc, list):
        return [serialize_doc(d) for d in doc]
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                new_doc[k] = str(v)
            elif isinstance(v, datetime):
                new_doc[k] = v.isoformat()
            elif isinstance(v, dict) or isinstance(v, list):
                new_doc[k] = serialize_doc(v)
            else:
                new_doc[k] = v
        return new_doc
    return doc

# Static file serving routes
@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    if os.path.exists(os.path.join(BASE_DIR, path)):
        return send_from_directory(BASE_DIR, path)
    return send_from_directory(BASE_DIR, "index.html")

# API: Health & Compass Connection Status
@app.route("/api/health", methods=["GET"])
def api_health():
    database = get_db()
    if database is not None:
        try:
            telemetry_count = database["telemetry_logs"].count_documents({})
            jobs_count = database["pipeline_jobs"].count_documents({})
            return jsonify({
                "status": "ONLINE",
                "mongodb_connected": True,
                "mongodb_uri": MONGO_URI,
                "database": DB_NAME,
                "collections": {
                    "telemetry_logs": telemetry_count,
                    "pipeline_jobs": jobs_count
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }), 200
        except Exception as e:
            return jsonify({
                "status": "ONLINE",
                "mongodb_connected": False,
                "error": str(e)
            }), 500
    else:
        return jsonify({
            "status": "ONLINE",
            "mongodb_connected": False,
            "message": "MongoDB not reachable on localhost:27017"
        }), 503

# API: Save Ingested Drone Video & Telemetry (12 Attributes)
@app.route("/api/telemetry", methods=["POST"])
def save_telemetry():
    database = get_db()
    if database is None:
        return jsonify({"error": "MongoDB unavailable"}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing payload"}), 400

        doc = {
            "source_label": data.get("source_label", "Unnamed Video"),
            "uploaded_at": datetime.utcnow(),
            "video_metadata": {
                "width": data.get("width"),
                "height": data.get("height"),
                "res_tag": data.get("res_tag"),
                "fps": data.get("fps"),
                "duration_sec": data.get("duration_sec")
            },
            "camera_intrinsics": data.get("camera_intrinsics", {}),
            "telemetry_12_attributes": {
                "timestamp_utc": data.get("timestamp_utc"),
                "timecode": data.get("timecode"),
                "latitude_deg": data.get("latitude_deg"),
                "latitude_dms": data.get("latitude_dms"),
                "longitude_deg": data.get("longitude_deg"),
                "longitude_dms": data.get("longitude_dms"),
                "altitude_agl_m": data.get("altitude_agl_m"),
                "altitude_msl_m": data.get("altitude_msl_m"),
                "barometric_alt_m": data.get("barometric_alt_m"),
                "video_fps_res": data.get("video_fps_res"),
                "camera_intrinsics_summary": data.get("camera_intrinsics_summary"),
                "yaw_deg": data.get("yaw_deg"),
                "pitch_deg": data.get("pitch_deg"),
                "roll_deg": data.get("roll_deg"),
                "imu_acceleration_mps2": data.get("imu_acc"),
                "imu_gyro_dps": data.get("imu_gyro"),
                "barometer_hpa": data.get("barometer_hpa"),
                "barometer_inhg": data.get("barometer_inhg"),
                "rtk_ppk_status": data.get("rtk_status"),
                "rtk_accuracy": data.get("rtk_accuracy"),
                "satellites_tracked": data.get("sat_count"),
                "ground_speed_mps": data.get("ground_speed_mps"),
                "velocity_vectors_mps": data.get("velocity_vectors")
            },
            "flight_trajectory_points": data.get("flight_trajectory_points", [])
        }

        result = database["telemetry_logs"].insert_one(doc)
        print(f"[MONGODB] Ingested telemetry doc ID: {result.inserted_id} for '{doc['source_label']}'")

        return jsonify({
            "success": True,
            "id": str(result.inserted_id),
            "database": DB_NAME,
            "collection": "telemetry_logs",
            "message": f"Saved flight telemetry for '{doc['source_label']}' in MongoDB Compass"
        }), 201

    except Exception as e:
        print(f"[MONGODB ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# API: Retrieve Recent Telemetry Records
@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    database = get_db()
    if database is None:
        return jsonify({"error": "MongoDB unavailable"}), 503

    try:
        limit = int(request.args.get("limit", 10))
        records = list(database["telemetry_logs"].find().sort("uploaded_at", -1).limit(limit))
        return jsonify({
            "success": True,
            "count": len(records),
            "records": serialize_doc(records)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Record 3D Reconstruction Pipeline Job Run
@app.route("/api/pipeline/job", methods=["POST"])
def save_pipeline_job():
    database = get_db()
    if database is None:
        return jsonify({"error": "MongoDB unavailable"}), 503

    try:
        data = request.get_json(force=True)
        job_doc = {
            "job_id": data.get("job_id", f"JOB-{int(datetime.utcnow().timestamp())}"),
            "created_at": datetime.utcnow(),
            "status": data.get("status", "COMPLETED"),
            "configuration": {
                "preset": data.get("preset", "balanced"),
                "frame_density": data.get("frame_density", 3),
                "dynamic_masking": data.get("dynamic_masking", True)
            },
            "reconstruction_metrics": {
                "keyframes_count": data.get("keyframes_count", 42),
                "gsd_accuracy_cm": data.get("gsd_accuracy_cm", 1.4),
                "vertices_count": data.get("vertices_count", 485210),
                "overlap_score_pct": data.get("overlap_score_pct", 99.4)
            },
            "deliverables": [
                {"format": "GLB", "type": "3D Textured Mesh Model", "status": "READY"},
                {"format": "LAS", "type": "Dense LiDAR Point Cloud", "status": "READY"},
                {"format": "OBJ", "type": "Wavefront CAD/Blender Mesh", "status": "READY"},
                {"format": "GEOTIFF", "type": "Orthomosaic GIS Map Layer", "status": "READY"}
            ],
            "execution_logs": data.get("execution_logs", [])
        }

        result = database["pipeline_jobs"].insert_one(job_doc)
        print(f"[MONGODB] Recorded pipeline job ID: {result.inserted_id}")

        return jsonify({
            "success": True,
            "id": str(result.inserted_id),
            "job_id": job_doc["job_id"],
            "database": DB_NAME,
            "collection": "pipeline_jobs",
            "message": "3D Pipeline Job successfully logged to MongoDB Compass"
        }), 201

    except Exception as e:
        print(f"[MONGODB ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# API: List Past Pipeline Jobs
@app.route("/api/pipeline/jobs", methods=["GET"])
def get_pipeline_jobs():
    database = get_db()
    if database is None:
        return jsonify({"error": "MongoDB unavailable"}), 503

    try:
        jobs = list(database["pipeline_jobs"].find().sort("created_at", -1).limit(10))
        return jsonify({
            "success": True,
            "count": len(jobs),
            "jobs": serialize_doc(jobs)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"==================================================")
    print(f" Aero3D OnePass Backend with MongoDB")
    print(f" Database: {DB_NAME} (mongodb://localhost:27017/)")
    print(f" Serving at: http://localhost:8000")
    print(f"==================================================")
    app.run(host="0.0.0.0", port=8000, debug=False)
