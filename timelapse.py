# timelapse.py
import os
import io
import requests
import concurrent.futures
from datetime import datetime
from dataclasses import dataclass
from PIL import Image, ImageOps
import numpy as np
import cv2
import dlib
from tqdm import tqdm
import logging


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
tqdm_handler = TqdmLoggingHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
tqdm_handler.setFormatter(formatter)
logger.addHandler(tqdm_handler)

face_predictor = None


@dataclass
class ProcessConfig:
    """
    Dataclass to hold configuration parameters for processing assets.
    """
    api_key: str
    base_url: str
    person_id: str
    output_folder: str = "output"
    padding_percent: float = 0.3
    resize_width: int = 512
    resize_height: int = 512
    min_face_width: int = 128
    min_face_height: int = 128
    pose_threshold: float = 25
    left_eye_pos: tuple = (0.35, 0.45)
    landmark_model_path: str = "shape_predictor_68_face_landmarks.dat"


def validate_immich_connection(api_key, base_url):
    """
    Validates that the provided Immich API key and base URL are working.

    Args:
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.

    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if not api_key or not base_url:
        return False, "API key and base URL are required."

    try:
        headers = {
            'Accept': 'application/json',
            'x-api-key': api_key,
        }
        # Try a simple ping to the server via the user endpoint
        url = f"{base_url}/server/about"
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            return True, "Connection successful."
        elif response.status_code == 401:
            return False, "Authentication failed. Invalid API key."
        else:
            return False, f"Server error: Status code {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, "Connection error. Check the base URL."
    except requests.exceptions.Timeout:
        return False, "Connection timed out. Server might be down."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def initialize_worker(landmark_model_path):
    """
    Initializes the face predictor in each worker process.
    """
    global face_predictor
    face_predictor = dlib.shape_predictor(landmark_model_path)


def get_assets_with_person(api_key, base_url, person_id, date_from=None, date_to=None):
    """
    Retrieve all image assets containing the specified person by querying the API.

    Args:
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.
        person_id (str): ID of the person to search for.
        date_from (str, optional): Start date in ISO format (YYYY-MM-DD).
        date_to (str, optional): End date in ISO format (YYYY-MM-DD).

    Returns:
        list: List of asset dictionaries.
    """
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

    if date_from:
        payload["takenAfter"] = f"{date_from}T00:00:00.000Z"

    if date_to:
        payload["takenBefore"] = f"{date_to}T23:59:59.999Z"

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
    """
    Downloads the original image asset from the API.

    Args:
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.
        asset_id (str): The asset's ID.

    Returns:
        bytes: The content of the downloaded image.
    """
    headers = {'x-api-key': api_key}
    response = requests.get(f'{base_url}/assets/{asset_id}/original', headers=headers)
    response.raise_for_status()
    return response.content


def crop_face_from_metadata(image, face_data, padding_percent):
    """
    Crops the face from the image using metadata and applies padding.

    Args:
        image (PIL.Image): The original image.
        face_data (dict): Metadata containing face bounding box info.
        padding_percent (float): Padding as a percentage of face dimensions.

    Returns:
        PIL.Image: The cropped face image.
    """
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
    """
    Estimates the head pose (pitch, yaw, roll) using facial landmarks.

    Args:
        shape (dlib.full_object_detection): Detected facial landmarks.
        img_size (tuple): The size of the image (width, height).

    Returns:
        tuple or None: (pitch, yaw, roll) in degrees if successful; otherwise None.
    """
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),  # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye left corner
        (shape.part(45).x, shape.part(45).y),  # Right eye right corner
        (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)  # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
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
    if not success:
        logger.info("Head pose estimation failed in solvePnP.")
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = [float(angle) for angle in euler_angles]
    return pitch, yaw, roll



def align_face(image, face_data, desired_face_width, desired_face_height, left_eye_pos, pose_threshold):
    """
    Aligns the face in the image using facial landmarks and head pose estimation.

    Args:
        image (PIL.Image): The cropped image containing just the face.
        face_data (dict): Metadata containing face bounding box info (not used for detection).
        desired_face_width (int): The desired output face width.
        desired_face_height (int): The desired output face height.
        left_eye_pos (tuple): The desired relative position of the left eye.
        pose_threshold (float): The maximum allowable head pose deviation.

    Returns:
        PIL.Image or None: The aligned face image if successful, otherwise None.
    """
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Since we already have a cropped face image, create a rectangle for the whole image
    img_height, img_width = gray.shape
    rect = dlib.rectangle(0, 0, img_width, img_height)

    # Get facial landmarks
    shape = face_predictor(gray, rect)

    # Check if face landmarks were detected
    if shape.num_parts != 68:
        logger.info("Landmark detection failed - could not find all 68 facial landmarks.")
        return None

    # Get head pose
    head_pose = get_head_pose(shape, (img_width, img_height))
    if head_pose is None:
        return None

    pitch, yaw, roll = head_pose
    if abs(abs(pitch) - 180) > pose_threshold or abs(yaw) > pose_threshold:
        logger.info(f"Face not frontal enough: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}. Discarding.")
        # return None

    # Convert landmarks to numpy array
    shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")

    # Calculate eye centers
    left_eye_center = shape_np[36:42].mean(axis=0).astype("int")
    right_eye_center = shape_np[42:48].mean(axis=0).astype("int")

    # Calculate angle and scale
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
    right_eye_pos = 1.0 - left_eye_pos[0]
    desired_eye_distance = (right_eye_pos - left_eye_pos[0]) * desired_face_width
    scale = desired_eye_distance / eye_distance

    # Calculate center of eyes
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                   (left_eye_center[1] + right_eye_center[1]) / 2.0)

    # Create transformation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Update translation component of the matrix
    tX = desired_face_width * 0.5
    tY = desired_face_height * left_eye_pos[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    # Apply transformation
    aligned_face_np = cv2.warpAffine(
        image_np,
        M,
        (desired_face_width, desired_face_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return Image.fromarray(aligned_face_np)


def process_asset_worker(asset, config: ProcessConfig):
    """
    Worker function to process a single asset.

    This function downloads the asset, crops the face based on metadata,
    verifies resolution, aligns the face, and then saves the aligned face.

    Args:
        asset (dict): The asset metadata.
        config (ProcessConfig): Configuration parameters.

    Returns:
        str or None: The file path of the saved image if processing is successful; otherwise None.
    """
    try:
        asset_id = asset['id']


        image_bytes = download_asset(config.api_key, config.base_url, asset_id)
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as e:
        logger.info(f"Error processing asset {asset.get('id')}: {e}")
        return None

    matching_person = next((p for p in asset.get('people', []) if p.get('id') == config.person_id), None)
    if not matching_person:
        logger.info("Subject not in image.")
        return None
    faces = matching_person.get('faces', [])
    if not faces:
        logger.info("No face data available.")
        return None
    face_data = faces[0]
    cropped_face = crop_face_from_metadata(image, face_data, config.padding_percent)
    face_width, face_height = cropped_face.size
    if face_width < config.min_face_width or face_height < config.min_face_height:
        logger.info(f"Face resolution too low ({face_width}x{face_height}).")
        return None

    # Pass face_data to align_face
    aligned_face = align_face(
        cropped_face,
        face_data,
        desired_face_width=config.resize_width,
        desired_face_height=config.resize_height,
        left_eye_pos=config.left_eye_pos,
        pose_threshold=config.pose_threshold
    )

    if aligned_face is None:
        return None
    os.makedirs(config.output_folder, exist_ok=True)
    dt = datetime.fromisoformat(asset['fileCreatedAt'].replace("Z", "+00:00"))
    timestamp = dt.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(config.output_folder, f"{timestamp}.jpg")
    aligned_face.save(filename)
    return filename


def process_faces(config: ProcessConfig, max_workers=1, progress_callback=None, date_from=None, date_to=None,
                  cancel_flag=None):
    """
    Processes assets containing the person and saves aligned face images.

    This function retrieves assets from the API, then uses a process pool to
    concurrently download, crop, and align faces.

    Args:
        config (ProcessConfig): Configuration parameters.
        max_workers (int): Number of worker processes.
        progress_callback (callable, optional): A callback function for progress updates.
        date_from (str, optional): Start date for filtering assets.
        date_to (str, optional): End date for filtering assets.
        cancel_flag (callable, optional): A function that returns True if processing should be cancelled.

    Returns:
        list: A list of file paths of the saved images.
    """
    os.makedirs(config.output_folder, exist_ok=True)

    if cancel_flag and cancel_flag():
        logger.info("Processing was cancelled.")
        return []

    assets = get_assets_with_person(config.api_key, config.base_url, config.person_id, date_from, date_to)
    logger.info(f"Found {len(assets)} assets containing the person.")

    total_assets = len(assets)
    if progress_callback:
        progress_callback(0, total_assets)
    processed_files = []
    completed_count = 0

    initializer_args = (config.landmark_model_path,)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initialize_worker,
            initargs=initializer_args) as executor:
        future_to_asset = {executor.submit(process_asset_worker, asset, config): asset
                           for asset in assets}
        for future in tqdm(concurrent.futures.as_completed(future_to_asset), total=total_assets):

            if cancel_flag and cancel_flag():
                logger.info("Processing was cancelled.")
                for f in future_to_asset:
                    f.cancel()
                executor.shutdown(wait=False)
                return processed_files

            try:
                result = future.result()
                if result is not None:
                    processed_files.append(result)
            except Exception as e:
                logger.info(f"Asset processing failed: {e}")

            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total_assets)

    logger.info(f"Finished processing. {len(processed_files)} images saved out of {total_assets} assets.")
    return processed_files