import os
import multiprocessing
import threading
import subprocess
from flask import Flask, request, render_template, jsonify, redirect, url_for
from timelapse import process_faces, ProcessConfig, validate_immich_connection
import logging
import uuid
from datetime import datetime

app = Flask(__name__)

# Critical parameters provided via environment variables
API_KEY = os.environ.get("IMMICH_API_KEY", "")
BASE_URL = os.environ.get("IMMICH_BASE_URL", "")
OUTPUT_FOLDER = "output"

# Model paths
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

AVAILABLE_CORES = multiprocessing.cpu_count()

# Global progress dictionary – only one job at a time is assumed here
progress_info = {"completed": 0, "total": 0, "status": "idle"}
# Global processing thread reference
processing_thread = None
# Global flag to signal cancellation
cancel_requested = False

class ProgressRouteFilter(logging.Filter):
    def filter(self, record):
        # Filter out logs containing the progress route
        return "/progress" not in record.getMessage()

log = logging.getLogger('werkzeug')
log.addFilter(ProgressRouteFilter())

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


def background_process(person_id, resize_size, face_resolution_threshold, pose_threshold,
                     left_eye_pos, output_folder, api_key, base_url, progress_callback=None, cancel_flag=None):
    """
    Background process to handle face alignment and timelapse creation.

    Args:
        person_id (str): ID of the person to process.
        resize_size (int): Size to resize the output images to.
        face_resolution_threshold (int): Minimum face resolution threshold.
        pose_threshold (float): Maximum allowed head pose deviation.
        left_eye_pos (tuple): Desired position of the left eye in the output.
        output_folder (str): Folder to save the output images.
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.
        progress_callback (callable, optional): Callback for progress updates.
        cancel_flag (callable, optional): Function to check if process should be cancelled.
    """
    try:
        config = ProcessConfig(
            api_key=api_key,
            base_url=base_url,
            person_id=person_id,
            output_folder=output_folder,
            resize_width=resize_size,
            resize_height=resize_size,
            min_face_width=face_resolution_threshold,
            min_face_height=face_resolution_threshold,
            pose_threshold=pose_threshold,
            left_eye_pos=left_eye_pos
        )

        # Process the faces
        processed_files = process_faces(
            config=config,
            max_workers=1,
            progress_callback=progress_callback,
            cancel_flag=cancel_flag
        )

        return processed_files

    except Exception as e:
        logger.error(f"Error in background process: {str(e)}")
        raise


def check_output_folder():
    """
    Checks if the output folder is empty.

    Returns:
        tuple: (is_empty, file_count) - Boolean indicating if folder is empty and number of files
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        return True, 0

    files = [f for f in os.listdir(OUTPUT_FOLDER) if os.path.isfile(os.path.join(OUTPUT_FOLDER, f))]
    return len(files) == 0, len(files)


@app.route("/progress")
def progress():
    """
    Endpoint to return current progress as JSON.
    """
    return jsonify(progress_info)


@app.route("/check-connection")
def check_connection():
    """
    Endpoint to check the Immich server connection.
    """
    is_valid, message = validate_immich_connection(API_KEY, BASE_URL)
    return jsonify({"valid": is_valid, "message": message})


@app.route("/cancel", methods=["POST"])
def cancel():
    """
    Endpoint to cancel the current processing job.
    """
    global processing_thread, cancel_requested

    # Set the cancel flag
    cancel_requested = True

    if processing_thread and processing_thread.is_alive():
        progress_info["status"] = "cancelled"
        return jsonify({"success": True, "message": "Processing cancelled."})
    else:
        cancel_requested = False
        return jsonify({"success": False, "message": "No active processing to cancel."})


@app.route("/process", methods=["POST"])
def process():
    """Handle the processing request."""
    try:
        person_id = request.form.get("person_id")
        if not person_id:
            return jsonify({"error": "Person ID is required"}), 400

        resize_size = int(request.form.get("resize_size", 512))
        face_resolution_threshold = int(request.form.get("face_resolution_threshold", 128))
        pose_threshold = float(request.form.get("pose_threshold", 25))
        left_eye_x = float(request.form.get("left_eye_x", 0.4))
        left_eye_y = float(request.form.get("left_eye_y", 0.4))
        left_eye_pos = (left_eye_x, left_eye_y)

        # Create output folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join("output", f"timelapse_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)

        # Start background process
        process_id = str(uuid.uuid4())
        process_info = {
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "output_folder": output_folder
        }
        active_processes[process_id] = process_info

        # Start the background process
        process = multiprocessing.Process(
            target=background_process,
            args=(person_id, resize_size, face_resolution_threshold, pose_threshold,
                  left_eye_pos, output_folder, API_KEY, BASE_URL)
        )
        process.start()
        active_processes[process_id]["process"] = process

        return jsonify({"process_id": process_id})

    except Exception as e:
        logger.error(f"Error in process route: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Index route that displays the form and starts processing in a background thread on POST.
    """
    global processing_thread, cancel_requested

    result = None
    error = None
    warning = None

    # Check if output folder is empty
    is_empty, file_count = check_output_folder()
    if not is_empty:
        warning = f"Output folder is not empty. Contains {file_count} files. New images will be added to this folder."

    # Check if Immich server connection is valid
    is_valid, message = validate_immich_connection(API_KEY, BASE_URL)
    if not is_valid and request.method == "POST":
        error = f"Immich server connection error: {message}"
        return render_template("index.html", error=error, warning=warning,
                               max_workers_options=list(range(1, AVAILABLE_CORES + 1)))

    if request.method == "POST":
        try:
            # Make sure any previous cancel request is cleared
            cancel_requested = False

            person_id = request.form["person_id"]
            resize_size = int(request.form.get("resize_size", 512))
            face_resolution_threshold = int(request.form.get("face_resolution_threshold", 128))
            pose_threshold = float(request.form.get("pose_threshold", 25))
            left_eye_x = float(request.form.get("left_eye_x", 0.4))
            left_eye_y = float(request.form.get("left_eye_y", 0.4))
            left_eye_pos = (left_eye_x, left_eye_y)

            # Date ranges are optional
            date_from = request.form.get("date_from") or None
            date_to = request.form.get("date_to") or None

            compile_video = request.form.get("compile_video") == "on"
            framerate = int(request.form.get("framerate", 24))

            # Reset progress info before starting
            progress_info["completed"] = 0
            progress_info["total"] = 0
            progress_info["status"] = "idle"
            progress_info.pop("video_path", None)

            # Start the processing in a background thread
            processing_thread = threading.Thread(
                target=background_process,
                args=(person_id, resize_size, face_resolution_threshold, pose_threshold,
                      left_eye_pos, OUTPUT_FOLDER, API_KEY, BASE_URL, update_progress, lambda: cancel_requested)
            )
            processing_thread.start()
            result = "Processing started. Please wait and watch the progress bar below."
        except Exception as e:
            error = f"Error processing request: {e}"

    return render_template("index.html", result=result, error=error, warning=warning,
                           max_workers_options=list(range(1, AVAILABLE_CORES + 1)))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)