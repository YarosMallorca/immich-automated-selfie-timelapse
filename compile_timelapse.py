import os
import subprocess
import logging
from typing import Callable, List
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
        image_folder: Path to folder containing timestamped JPEG images
        output_path: Path where the output video should be saved
        framerate: Desired frames per second
        update_progress: Callback function to report progress (current: int, total: int)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get list of image files and sort by timestamp
        image_files = sorted([
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg'))
        ])
        
        if not image_files:
            logger.error("No image files found in folder")
            update_progress(0, 0)
            return False
            
        total_frames = len(image_files)
        logger.info(f"Found {total_frames} images to process")
        
        # Create ffmpeg input file list
        input_file = os.path.join(image_folder, "input.txt")
        frame_duration = 1.0 / framerate  # Calculate duration based on requested framerate
        
        with open(input_file, "w") as f:
            for img in image_files:
                f.write(f"file '{os.path.join(image_folder, img)}'\n")
                f.write(f"duration {frame_duration}\n")
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", input_file,
            "-framerate", str(framerate),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",  # Overwrite output file if it exists
            output_path
        ]
        
        # Run ffmpeg with progress reporting
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor progress
        frame_count = 0
        for line in process.stderr:
            if "frame=" in line:
                try:
                    # Extract frame number from ffmpeg output
                    frame = int(line.split("frame=")[1].split()[0])
                    frame_count = min(frame, total_frames)  # Ensure we don't exceed total
                    update_progress(frame_count, total_frames)
                except (IndexError, ValueError):
                    continue
                    
        # Wait for process to complete
        process.wait()
        
        # Clean up temporary file
        os.remove(input_file)
        
        if process.returncode == 0:
            update_progress(total_frames, total_frames)
            return True
        else:
            logger.error(f"ffmpeg failed with return code {process.returncode}")
            update_progress(0, total_frames)
            return False
            
    except Exception as e:
        logger.error(f"Error compiling timelapse: {str(e)}")
        update_progress(0, total_frames if 'total_frames' in locals() else 0)
        return False 