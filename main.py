import os
import multiprocessing
from flask import Flask, request, render_template
from timelapse import process_faces

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

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    # Create max_workers_options as a list of numbers from 1 to AVAILABLE_CORES
    max_workers_options = list(range(1, AVAILABLE_CORES + 1))
    if request.method == "POST":
        try:
            api_key = API_KEY
            base_url = BASE_URL
            output_folder = OUTPUT_FOLDER

            # Get user input from the form
            person_id = request.form["person_id"]
            padding_percent = float(request.form.get("padding_percent", 30)) / 100
            resize_size = int(request.form.get("resize_size", 512))
            face_resolution_threshold = int(request.form.get("face_resolution_threshold", 128))
            pose_threshold = float(request.form.get("pose_threshold", 25))
            max_workers = int(request.form.get("max_workers", 1))  # default is 1

            processed_files = process_faces(
                api_key=api_key,
                base_url=base_url,
                person_id=person_id,
                output_folder=output_folder,
                padding_percent=padding_percent,
                resize_width=resize_size,
                resize_height=resize_size,
                min_face_width=face_resolution_threshold,
                min_face_height=face_resolution_threshold,
                pose_threshold=pose_threshold,
                desired_left_eye=LEFT_EYE_POS,
                max_workers=max_workers,
                face_detect_model_path=FACE_DETECT_MODEL,
                landmark_model_path=LANDMARK_MODEL
            )
            result = f"Finished processing. {len(processed_files)} images saved in '{output_folder}'."
        except Exception as e:
            error = f"Error processing request: {e}"
    return render_template("index.html", result=result, error=error, max_workers_options=max_workers_options)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
