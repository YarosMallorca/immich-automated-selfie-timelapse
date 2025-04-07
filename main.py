import os
import multiprocessing
import threading
from flask import Flask, request, render_template, jsonify
from timelapse import process_faces, ProcessConfig

app = Flask(__name__)

# Critical parameters provided via environment variables
API_KEY = os.environ.get("IMMICH_API_KEY", "")
BASE_URL = os.environ.get("IMMICH_BASE_URL", "")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "output")

# Model paths
FACE_DETECT_MODEL = "mmod_human_face_detector.dat"
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

LEFT_EYE_POS = (0.35, 0.45)
AVAILABLE_CORES = multiprocessing.cpu_count()

# Global progress dictionary – only one job at a time is assumed here
progress_info = {"completed": 0, "total": 0, "status": "idle"}


def update_progress(current, total):
    """
    Updates the global progress dictionary.

    Args:
        current (int): Number of completed tasks.
        total (int): Total number of tasks.
    """
    progress_info["completed"] = current
    progress_info["total"] = total
    progress_info["status"] = "running" if current < total else "done"


def background_process(person_id, padding_percent, resize_size, face_resolution_threshold, pose_threshold, max_workers):
    """
    Background process that creates a configuration object and calls process_faces.

    Args:
        person_id (str): The target person ID.
        padding_percent (float): Padding percentage for face cropping.
        resize_size (int): Desired width and height for the aligned face image.
        face_resolution_threshold (int): Minimum required face resolution.
        pose_threshold (float): Maximum allowed head pose deviation.
        max_workers (int): Number of concurrent worker processes.
    """
    try:
        progress_info["status"] = "running"
        # Build the configuration object for processing
        config = ProcessConfig(
            api_key=API_KEY,
            base_url=BASE_URL,
            person_id=person_id,
            output_folder=OUTPUT_FOLDER,
            padding_percent=padding_percent,
            resize_width=resize_size,
            resize_height=resize_size,
            min_face_width=face_resolution_threshold,
            min_face_height=face_resolution_threshold,
            pose_threshold=pose_threshold,
            desired_left_eye=LEFT_EYE_POS,
            face_detect_model_path=FACE_DETECT_MODEL,
            landmark_model_path=LANDMARK_MODEL
        )
        process_faces(config, max_workers=max_workers, progress_callback=update_progress)
    except Exception as e:
        progress_info["status"] = f"error: {e}"
    else:
        progress_info["status"] = "done"


@app.route("/progress")
def progress():
    """
    Endpoint to return current progress as JSON.
    """
    return jsonify(progress_info)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Index route that displays the form and starts processing in a background thread on POST.
    """
    result = None
    error = None
    # Create max_workers_options as a list from 1 to AVAILABLE_CORES
    max_workers_options = list(range(1, AVAILABLE_CORES + 1))
    if request.method == "POST":
        try:
            person_id = request.form["person_id"]
            padding_percent = float(request.form.get("padding_percent", 30)) / 100
            resize_size = int(request.form.get("resize_size", 512))
            face_resolution_threshold = int(request.form.get("face_resolution_threshold", 128))
            pose_threshold = float(request.form.get("pose_threshold", 25))
            max_workers = int(request.form.get("max_workers", 1))  # default is 1

            # Reset progress info before starting
            progress_info["completed"] = 0
            progress_info["total"] = 0
            progress_info["status"] = "idle"

            # Start the processing in a background thread
            threading.Thread(
                target=background_process,
                args=(person_id, padding_percent, resize_size, face_resolution_threshold, pose_threshold, max_workers)
            ).start()
            result = "Processing started. Please wait and watch the progress bar below."
        except Exception as e:
            error = f"Error processing request: {e}"
    return render_template("index.html", result=result, error=error, max_workers_options=max_workers_options)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
