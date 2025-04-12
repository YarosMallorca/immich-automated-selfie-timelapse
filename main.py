import logging
import multiprocessing
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request
from timelapse import ProcessConfig, process_faces, validate_immich_connection

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Filter out progress route logs
class ProgressRouteFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/progress" not in record.getMessage()

log = logging.getLogger('werkzeug')
log.addFilter(ProgressRouteFilter())

@dataclass
class AppConfig:
    """Configuration for the application."""
    api_key: str = os.environ.get("IMMICH_API_KEY", "")
    base_url: str = os.environ.get("IMMICH_BASE_URL", "")
    output_folder: str = "output"
    landmark_model: str ="shape_predictor_68_face_landmarks.dat"
    default_resize_size: int = 512
    default_face_resolution_threshold: int = 128
    default_pose_threshold: float = 25.0
    default_left_eye_pos: Tuple[float, float] = (0.35, 0.4)
    default_framerate: int = 24
    default_date_format: str = "%Y-%m-%d"



# Initialize Flask app
app = Flask(__name__)

# Global state
AVAILABLE_CORES = multiprocessing.cpu_count()
progress_info: Dict[str, any] = {"completed": 0, "total": 0, "status": "idle"}
processing_thread: Optional[threading.Thread] = None
cancel_requested: bool = False
config = AppConfig()

def update_progress(current: int, total: int) -> None:
    """Update the global progress information.
    
    Args:
        current: Number of completed tasks
        total: Total number of tasks
    """
    progress_info["completed"] = current
    progress_info["total"] = total
    progress_info["status"] = "running" if current < total else "done"

def check_output_folder() -> Tuple[bool, int]:
    """Check if the output folder is empty.
    
    Returns:
        Tuple containing (is_empty, file_count)
    """
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder, exist_ok=True)
        return True, 0

    files = [f for f in os.listdir(config.output_folder) 
             if os.path.isfile(os.path.join(config.output_folder, f))]
    return len(files) == 0, len(files)

def background_process(
    person_id: str,
    resize_size: int,
    face_resolution_threshold: int,
    pose_threshold: float,
    left_eye_pos: Tuple[float, float],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    cancel_flag: Optional[Callable] = None
) -> List[str]:
    """Process faces in the background.
    
    Args:
        person_id: ID of the person to process
        resize_size: Size to resize output images to
        face_resolution_threshold: Minimum face resolution threshold
        pose_threshold: Maximum allowed head pose deviation
        left_eye_pos: Desired position of the left eye in output
        date_from: Optional start date in YYYY-MM-DD format
        date_to: Optional end date in YYYY-MM-DD format
        progress_callback: Optional callback for progress updates
        cancel_flag: Optional function to check for cancellation
        
    Returns:
        List of processed file paths
    """
    try:
        process_config = ProcessConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            person_id=person_id,
            output_folder=config.output_folder,
            resize_width=resize_size,
            resize_height=resize_size,
            min_face_width=face_resolution_threshold,
            min_face_height=face_resolution_threshold,
            pose_threshold=pose_threshold,
            left_eye_pos=left_eye_pos,
            landmark_model_path=config.landmark_model,
            date_from=date_from,
            date_to=date_to
        )

        return process_faces(
            config=process_config,
            max_workers=1,
            progress_callback=progress_callback,
            cancel_flag=cancel_flag
        )

    except Exception as e:
        logger.error(f"Error in background process: {str(e)}")
        raise

@app.route("/progress")
def progress() -> Dict[str, any]:
    """Get current progress information."""
    return jsonify(progress_info)

@app.route("/check-connection")
def check_connection() -> Dict[str, any]:
    """Check connection to Immich server."""
    is_valid, message = validate_immich_connection(config.api_key, config.base_url)
    return jsonify({"valid": is_valid, "message": message})

@app.route("/cancel", methods=["POST"])
def cancel() -> Dict[str, any]:
    """Cancel the current processing job."""
    global processing_thread, cancel_requested

    cancel_requested = True

    if processing_thread and processing_thread.is_alive():
        progress_info["status"] = "cancelled"
        return jsonify({"success": True, "message": "Processing cancelled."})
    
    cancel_requested = False
    return jsonify({"success": False, "message": "No active processing to cancel."})

@app.route("/", methods=["GET", "POST"])
def index() -> str:
    """Handle the main page and processing requests."""
    global processing_thread, cancel_requested

    result = None
    error = None
    warning = None

    # Check output folder status
    is_empty, file_count = check_output_folder()
    if not is_empty:
        warning = f"Output folder is not empty. Contains {file_count} files. New images will be added to this folder."

    # Validate connection on POST
    if request.method == "POST":
        is_valid, message = validate_immich_connection(config.api_key, config.base_url)
        if not is_valid:
            error = f"Immich server connection error: {message}"
            return render_template("index.html", error=error, warning=warning,
                                max_workers_options=list(range(1, AVAILABLE_CORES + 1)))

        try:
            cancel_requested = False

            # Get form data with defaults
            person_id = request.form["person_id"]
            resize_size = int(request.form.get("resize_size", config.default_resize_size))
            face_resolution_threshold = int(request.form.get("face_resolution_threshold", 
                                                          config.default_face_resolution_threshold))
            pose_threshold = float(request.form.get("pose_threshold", config.default_pose_threshold))

            # Optional date ranges
            date_from = request.form.get("date_from") or None
            date_to = request.form.get("date_to") or None

            # Video compilation options
            compile_video = request.form.get("compile_video") == "on"
            framerate = int(request.form.get("framerate", config.default_framerate))

            # Reset progress info
            progress_info.update({
                "completed": 0,
                "total": 0,
                "status": "idle"
            })
            progress_info.pop("video_path", None)

            # Start processing
            processing_thread = threading.Thread(
                target=background_process,
                args=(person_id, resize_size, face_resolution_threshold, pose_threshold,
                      config.default_left_eye_pos, config.output_folder, date_from, date_to,
                      update_progress, lambda: cancel_requested)
            )
            processing_thread.start()
            result = "Processing started. Please wait and watch the progress bar below."

        except Exception as e:
            error = f"Error processing request: {e}"

    return render_template("index.html", 
                         result=result, 
                         error=error, 
                         warning=warning,
                         max_workers_options=list(range(1, AVAILABLE_CORES + 1)))

if __name__ == "__main__":
    app.run(debug=True)