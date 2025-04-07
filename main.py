import os
import multiprocessing
import threading
import subprocess
from flask import Flask, request, render_template, jsonify, redirect, url_for
from timelapse import process_faces, ProcessConfig, validate_immich_connection

app = Flask(__name__)

# Critical parameters provided via environment variables
API_KEY = os.environ.get("IMMICH_API_KEY", "")
BASE_URL = os.environ.get("IMMICH_BASE_URL", "")
OUTPUT_FOLDER = "output"

# Model paths
FACE_DETECT_MODEL = "mmod_human_face_detector.dat"
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

LEFT_EYE_POS = (0.35, 0.45)
AVAILABLE_CORES = multiprocessing.cpu_count()

# Global progress dictionary – only one job at a time is assumed here
progress_info = {"completed": 0, "total": 0, "status": "idle"}
# Global processing thread reference
processing_thread = None
# Global flag to signal cancellation
cancel_requested = False


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


def background_process(person_id, padding_percent, resize_size, face_resolution_threshold, pose_threshold, max_workers,
                       date_from, date_to, compile_video, framerate):
    """
    Background process that creates a configuration object and calls process_faces.

    Args:
        person_id (str): The target person ID.
        padding_percent (float): Padding percentage for face cropping.
        resize_size (int): Desired width and height for the aligned face image.
        face_resolution_threshold (int): Minimum required face resolution.
        pose_threshold (float): Maximum allowed head pose deviation.
        max_workers (int): Number of concurrent worker processes.
        date_from (str): Start date for asset filtering.
        date_to (str): End date for asset filtering.
        compile_video (bool): Whether to compile the images into a video.
        framerate (int): Frames per second for the output video.
    """
    global cancel_requested
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

        # Pass the cancel flag to the process_faces function
        processed_files = process_faces(config, max_workers=max_workers, progress_callback=update_progress,
                                        date_from=date_from, date_to=date_to, cancel_flag=lambda: cancel_requested)

        # Check if processing was cancelled
        if cancel_requested:
            progress_info["status"] = "cancelled"
            cancel_requested = False
            return

        # Compile video if requested and there are processed files
        if compile_video and processed_files:
            output_video = os.path.join(OUTPUT_FOLDER, "timelapse.mp4")
            ffmpeg_command = [
                "ffmpeg", "-y",
                "-framerate", str(framerate),
                "-pattern_type", "glob",
                "-i", os.path.join(OUTPUT_FOLDER, "*.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video
            ]
            try:
                subprocess.run(ffmpeg_command, check=True)
                progress_info["video_path"] = output_video
                progress_info["status"] = "video_done"
            except subprocess.CalledProcessError as e:
                progress_info["status"] = f"Video compilation failed: {e}"
        else:
            progress_info["status"] = "done"
    except Exception as e:
        progress_info["status"] = f"error: {e}"


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
            padding_percent = float(request.form.get("padding_percent", 30)) / 100
            resize_size = int(request.form.get("resize_size", 512))
            face_resolution_threshold = int(request.form.get("face_resolution_threshold", 128))
            pose_threshold = float(request.form.get("pose_threshold", 25))
            max_workers = int(request.form.get("max_workers", 1))

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
                args=(person_id, padding_percent, resize_size, face_resolution_threshold, pose_threshold, max_workers,
                      date_from, date_to, compile_video, framerate)
            )
            processing_thread.start()
            result = "Processing started. Please wait and watch the progress bar below."
        except Exception as e:
            error = f"Error processing request: {e}"

    return render_template("index.html", result=result, error=error, warning=warning,
                           max_workers_options=list(range(1, AVAILABLE_CORES + 1)))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)