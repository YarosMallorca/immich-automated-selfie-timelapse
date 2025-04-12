import os
import subprocess
import logging
import tempfile
import shutil
from typing import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

def compile_timelapse(
    image_folder: str,
    output_path: str,
    framerate: int,
    update_progress: Callable[[int, int], None]
) -> bool:
    """
    Compile a timelapse video from a folder of images using ffmpeg.
    
    Args:
        image_folder: Path to folder containing timestamped JPEG images.
        output_path: Path where the output video should be saved.
        framerate: Desired frames per second.
        update_progress: Callback function to report progress (current: int, total: int).
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        # List and sort JPEG image files in the folder.
        image_files = sorted([
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg'))
        ])
        
        total_frames = len(image_files)
        if total_frames == 0:
            logger.error("No image files found in folder")
            update_progress(0, 0)
            return False
        
        logger.info(f"Found {total_frames} images to process")
        
        # Create a temporary directory to hold sequential files.
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")
        
        try:
            for idx, img in enumerate(image_files):
                src = os.path.join(image_folder, img)
                dst = os.path.join(temp_dir, f"{idx:06d}.jpg")
                try:
                    os.symlink(src, dst)
                except (AttributeError, NotImplementedError, OSError):
                    # Fallback to copying if symlinking is not supported.
                    shutil.copy2(src, dst)
            
            input_pattern = os.path.join(temp_dir, "%06d.jpg")
            cmd = [
                "ffmpeg",
                "-framerate", str(framerate),
                "-i", input_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-y",  # Overwrite output file if it exists
                output_path
            ]
            logger.info("Running ffmpeg command: " + " ".join(cmd))
            
            # Run ffmpeg and capture the stderr for progress updates.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            frame_count = 0
            for line in process.stderr:
                if "frame=" in line:
                    try:
                        frame = int(line.split("frame=")[1].split()[0])
                        frame_count = min(frame, total_frames)
                        update_progress(frame_count, total_frames)
                    except (IndexError, ValueError):
                        continue
            
            process.wait()
            
            if process.returncode == 0:
                update_progress(total_frames, total_frames)
                logger.info("Timelapse video created successfully")
                return True
            else:
                logger.error(f"ffmpeg failed with return code {process.returncode}")
                update_progress(0, total_frames)
                return False
        
        finally:
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
    
    except Exception as e:
        logger.error(f"Error compiling timelapse: {str(e)}")
        update_progress(0, total_frames if 'total_frames' in locals() else 0)
        return False
