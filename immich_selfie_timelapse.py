#!/usr/bin/env python3
"""
This tool helps create selfie timelapses from your Immich instance.
It uses the powerful machine learning features of Immich to gather all the photographs where a particular individual
appears, retrieves the bounding box metadata, and automatically crops and aligns the photos.
Some manual sorting is still required to achieve the best effect in the video.
I personally found that a video frame rate of 15 fps looks pretty good.

Script by Arnaud Cayrol
"""


import os
import io
import requests
import concurrent.futures
import argparse
from datetime import datetime

from PIL import Image, ImageOps
import numpy as np
import cv2
import dlib
from tqdm import tqdm
import logging

# Custom logging handler that works with tqdm
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
tqdm_handler = TqdmLoggingHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
tqdm_handler.setFormatter(formatter)
logger.addHandler(tqdm_handler)


def get_assets_with_person(api_key, base_url, person_id):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': api_key,
    }
    url = f"{base_url}/search/metadata"
    all_assets = []
    payload = {
        "page": 1,
        "type": "IMAGE",
        "personIds": [person_id],
        "withArchived": False,
        "withDeleted": True,
        "withExif": True,
        "withPeople": True,
        "withStacked": True,
    }
    while payload["page"] is not None:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logger.info(f"Error fetching page {payload['page']}: {response.status_code} - {response.text}")
            break
        data = response.json()
        if not data:
            break
        all_assets.extend(data['assets']['items'])
        logger.info(f"Fetched page {payload['page']} with {len(data['assets']['items'])} assets")
        payload["page"] = data['assets'].get('nextPage')
    return all_assets


def download_asset(api_key, base_url, asset_id):
    headers = {'x-api-key': api_key}
    response = requests.get(f'{base_url}/assets/{asset_id}/original', headers=headers)
    response.raise_for_status()
    return response.content


def format_timestamp(timestamp):
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%d_%H%M%S")


def crop_face_from_metadata(image, face_data, padding_percent):
    face_img_width = face_data.get("imageWidth")
    face_img_height = face_data.get("imageHeight")
    img_width, img_height = image.size
    scale_x = img_width / face_img_width
    scale_y = img_height / face_img_height
    x1 = int(face_data.get("boundingBoxX1", 0) * scale_x)
    x2 = int(face_data.get("boundingBoxX2", 0) * scale_x)
    y1 = int(face_data.get("boundingBoxY1", 0) * scale_y)
    y2 = int(face_data.get("boundingBoxY2", 0) * scale_y)
    w = x2 - x1
    h = y2 - y1
    padding = int(max(w, h) * padding_percent)
    new_x1 = max(x1 - padding, 0)
    new_y1 = max(y1 - padding, 0)
    new_x2 = min(x2 + padding, img_width)
    new_y2 = min(y2 + padding, img_height)
    return image.crop((new_x1, new_y1, new_x2, new_y2))


def get_head_pose(shape, img_size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),
        (shape.part(8).x, shape.part(8).y),
        (shape.part(36).x, shape.part(36).y),
        (shape.part(45).x, shape.part(45).y),
        (shape.part(48).x, shape.part(48).y),
        (shape.part(54).x, shape.part(54).y)
    ], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    w, h = img_size
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_mat, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = [float(angle) for angle in eulerAngles]
    return pitch, yaw, roll


def align_face(image, predictor, detector, desired_face_width, desired_face_height,
               desired_left_eye, pose_threshold):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Detect faces using the provided detector.
    detections = detector(gray)
    if not detections:
        logger.info("No face detected in the crop. Discarding.")
        return None

    # If using CNN detector, extract the rectangle from the detection.
    if hasattr(detections[0], "rect"):
        rect = detections[0].rect
    else:
        rect = detections[0]

    shape = predictor(gray, rect)
    img_size = (image_np.shape[1], image_np.shape[0])
    pitch, yaw, roll = get_head_pose(shape, img_size)
    if abs(abs(pitch) - 180) > pose_threshold or abs(yaw) > pose_threshold:
        logger.info(f"Face not frontal enough: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}. Discarding.")
        return None
    shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")
    left_eye_center = shape_np[36:42].mean(axis=0).astype("int")
    right_eye_center = shape_np[42:48].mean(axis=0).astype("int")
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
    desired_right_eye_x = 1.0 - desired_left_eye[0]
    desired_eye_distance = (desired_right_eye_x - desired_left_eye[0]) * desired_face_width
    scale = desired_eye_distance / eye_distance
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                   (left_eye_center[1] + right_eye_center[1]) / 2.0)
    adjusted_scale = scale * 0.8
    M = cv2.getRotationMatrix2D(eyes_center, angle, adjusted_scale)
    extra_offset_x = 10
    tX = desired_face_width * 0.5 + extra_offset_x
    tY = desired_face_height * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    aligned_face_np = cv2.warpAffine(
        image_np,
        M,
        (desired_face_width, desired_face_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(aligned_face_np)


def process_asset_worker(asset, api_key, base_url, person_id, output_folder,
                         padding_percent, min_face_width, min_face_height,
                         resize_width, resize_height, pose_threshold, desired_left_eye,
                         cnn_model_path, predictor_model_path):

    detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
    # Use predictor_model_path from command line instead of hard-coded path.
    local_predictor = dlib.shape_predictor(predictor_model_path)
    try:
        asset_id = asset['id']
        timestamp = format_timestamp(asset['fileCreatedAt'])
        image_bytes = download_asset(api_key, base_url, asset_id)
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as e:
        logger.info(f"Error processing asset {asset.get('id')}: {e}")
        return None
    matching_person = next((p for p in asset.get('people', []) if p.get('id') == person_id), None)
    if not matching_person:
        logger.info("Subject not in image.")
        return None
    faces = matching_person.get('faces', [])
    if not faces:
        logger.info("No face data available.")
        return None
    face_data = faces[0]
    cropped_face = crop_face_from_metadata(image, face_data, padding_percent)
    face_width, face_height = cropped_face.size
    if face_width < min_face_width or face_height < min_face_height:
        logger.info(f"Face resolution too low ({face_width}x{face_height}).")
        return None
    aligned_face = align_face(cropped_face, local_predictor, detector,
                              desired_face_width=resize_width,
                              desired_face_height=resize_height,
                              desired_left_eye=desired_left_eye,
                              pose_threshold=pose_threshold)
    if aligned_face is None:
        return None
    filename = os.path.join(output_folder, f"{timestamp}.jpg")
    aligned_face.save(filename)
    return filename


def process_asset_wrapper(asset, process_args):
    return process_asset_worker(asset, *process_args)


def main():
    parser = argparse.ArgumentParser(description="Process and align faces from assets.")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--base-url", required=True, help="Base URL for the API")
    parser.add_argument("--person-id", required=True, help="ID of the person to search for")
    parser.add_argument("--output-folder", default="output", help="Folder to save output images")
    parser.add_argument("--padding-percent", type=float, default=0.3, help="Padding percentage for face crop")
    parser.add_argument("--resize-width", type=int, default=512, help="Output image width")
    parser.add_argument("--resize-height", type=int, default=512, help="Output image height")
    parser.add_argument("--min-face-width", type=int, default=128, help="Minimum face width")
    parser.add_argument("--min-face-height", type=int, default=128, help="Minimum face height")
    parser.add_argument("--pose-threshold", type=float, default=25, help="Threshold for acceptable head orientation towards camera")
    parser.add_argument("--desired-left-eye", type=float, nargs=2, default=[0.35, 0.45],
                        help="Desired left eye position as fraction (x y) in the output image")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--face-detect-model-path", default="mmod_human_face_detector.dat", help="Path to the CNN face detector model file")
    parser.add_argument("--landmark-model-path", default="shape_predictor_68_face_landmarks.dat", help="Path to the face landmark predictor model file")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    assets = get_assets_with_person(args.api_key, args.base_url, args.person_id)
    logger.info(f"Found {len(assets)} assets containing the person.")

    process_args = (
        args.api_key, args.base_url, args.person_id, args.output_folder,
        args.padding_percent, args.min_face_width, args.min_face_height,
        args.resize_width, args.resize_height, args.pose_threshold, tuple(args.desired_left_eye),
        args.face_detect_model_path, args.landmark_model_path
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(tqdm(
            executor.map(process_asset_wrapper, assets, [process_args] * len(assets)),
            total=len(assets)
        ))
    processed_files = [r for r in results if r is not None]
    logger.info(f"Finished processing. {len(processed_files)} images saved.")


if __name__ == "__main__":
    main()
