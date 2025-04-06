# main.py

from flask import Flask, request, render_template
from immich_selfie_timelapse import process_faces

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        # Read form fields, converting as necessary.
        api_key = request.form["api_key"]
        base_url = request.form["base_url"]
        person_id = request.form["person_id"]
        output_folder = request.form.get("output_folder", "output")
        padding_percent = float(request.form.get("padding_percent", 0.3))
        resize_width = int(request.form.get("resize_width", 512))
        resize_height = int(request.form.get("resize_height", 512))
        min_face_width = int(request.form.get("min_face_width", 128))
        min_face_height = int(request.form.get("min_face_height", 128))
        pose_threshold = float(request.form.get("pose_threshold", 25))
        desired_left_eye_x = float(request.form.get("desired_left_eye_x", 0.35))
        desired_left_eye_y = float(request.form.get("desired_left_eye_y", 0.45))
        max_workers = int(request.form.get("max_workers", 4))
        face_detect_model_path = request.form.get("face_detect_model_path", "mmod_human_face_detector.dat")
        landmark_model_path = request.form.get("landmark_model_path", "shape_predictor_68_face_landmarks.dat")

        # Run the process_faces function with provided parameters
        processed_files = process_faces(
            api_key=api_key,
            base_url=base_url,
            person_id=person_id,
            output_folder=output_folder,
            padding_percent=padding_percent,
            resize_width=resize_width,
            resize_height=resize_height,
            min_face_width=min_face_width,
            min_face_height=min_face_height,
            pose_threshold=pose_threshold,
            desired_left_eye=(desired_left_eye_x, desired_left_eye_y),
            max_workers=max_workers,
            face_detect_model_path=face_detect_model_path,
            landmark_model_path=landmark_model_path
        )
        result = f"Finished processing. {len(processed_files)} images saved in {output_folder}"
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
