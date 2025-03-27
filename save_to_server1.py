import subprocess
import requests
import json
import os
import tempfile
import shutil
from datetime import datetime
import re
import concurrent.futures
import cv2
import time

# URLs for APIs
UPLOAD_URL = 'http://192.168.30.35:2222/api/upload/multiple/files'
INCIDENT_URL = 'http://192.168.30.35:2222/process/vehicle/incident/data'
# === MULTIPLE CAMERAS CONFIG ===
CAMERA_CONFIGS = [
    # {
    #     "camera_id": 1,
    #     "rtsp_url": "rtsp://admin:Admin@123@192.168.20.201:554",
    #     "camera_location": "",
    #     "chainage": ""
    # },
    # {
    #     "camera_id": 2,
    #     "rtsp_url": "rtsp://admin:Admin@123@192.168.20.202:554",
    #     "camera_location": "",
    #     "chainage": ""
    # },
    {
        "camera_id": 3,
        "rtsp_url": "rtsp://admin:9990@192.168.80.25:554/profile1",
        "camera_location": "Udyog Vihar, Gurugram",
        "chainage": "+0.5 km",
      
    },
]

# For MP4 testing (if needed)
MP4_SOURCES = [
    {
        "camera_id": 3,
        "mp4_path": "/home/kaustav/AIML/HAZARD/car.mp4",
        "camera_location": "Watsoo Express, Gurugram",
        "chainage": "",
        "lane_directions": {
            "lane1": {
                "region": (0, 0, 320, 640),   # (x1, y1, x2, y2): region for lane 1
                "direction": (0, 1)           # Expected movement to the right
            },
            "lane2": {
                "region": (321, 0, 640, 640), # (x1, y1, x2, y2): region for lane 2
                "direction": (0, -1)          # Expected movement to the left
            }
        }
    }
]


def extract_ip(rtsp_url):
    """ Extracts IP address from RTSP URL. """
    match = re.search(r"(\d+\.\d+\.\d+\.\d+)", rtsp_url)
    return match.group(1) if match else "IP Not Found"


def create_video_evidence(frames, camera_id, incident_type, fps=15, duration=4, suffix=""):
    """
    Create a video evidence clip from a list of frames using FFmpeg with H.264 encoding.
    
    Args:
        frames (list): List of annotated frame images (numpy arrays in BGR format).
        camera_id (str or int): Identifier for the camera.
        incident_type (str): The incident type (used for naming the file).
        fps (int, optional): Frames per second for the output video. Defaults to 15.
        duration (int, optional): Duration (in seconds) of the video clip. Defaults to 2.
        suffix (str, optional): Optional suffix for the filename. Defaults to "".
        
    Returns:
        str or None: The filepath of the saved video evidence if successful, else None.
    """
    try:
        if not frames:
            print("No frames provided for video evidence.")
            return None
        
        # Calculate required frames for the specified duration.
        required_frames = fps * duration
        if len(frames) < required_frames:
            last_frame = frames[-1]
            frames.extend([last_frame] * (required_frames - len(frames)))
        else:
            frames = frames[:required_frames]
            
        evidence_dir = os.path.join("evidence", f"camera_{camera_id}")
        os.makedirs(evidence_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(evidence_dir, f"{incident_type}_{timestamp}{suffix}.mp4")
        
        # Write frames to a temporary directory as PNG images.
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                if not cv2.imwrite(frame_path, frame):
                    print(f"Warning: Failed to write frame {i} to {frame_path}")

            # Build the ffmpeg command for H.264 encoding.
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                video_filename
            ]
            print("Running FFmpeg command:")
            print(" ".join(cmd))
            
            # Run ffmpeg command.
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return None
            else:
                print(f"Video evidence successfully created: {video_filename}")
        
        return video_filename
        
    except Exception as e:
        print(f"Error creating video evidence for camera {camera_id}: {e}")
        return None


def convert_to_http_files(file_paths):
    """ Uploads files and returns HTTP URLs. Closes files immediately after upload to save memory. """
    file_urls = []

    try:
        files = [('file', open(file, 'rb')) for file in file_paths]
        response = requests.post(UPLOAD_URL, files=files)

        # Close file handles immediately after upload
        for _, f in files:
            f.close()

        if response.status_code == 200:
            file_urls = response.json().get('data', [])
            print(f"Files uploaded successfully: \n\n{file_urls}")
        else:
            print(f"⚠️ Upload failed: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error in file upload: {e}")

    # Delete local files after upload to free memory
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)

    return file_urls


def send_vehicle_incident_data(vehicle_number, camera_location, chainage, category, incident_type, description, confidence,
                               video_urls, images_urls, camera_id, rtsp, timestamp):
    """ Sends structured vehicle incident data to the server. """
    headers = {'Content-Type': 'application/json'}

    # Extract the camera IP
    camera_ip = extract_ip(rtsp)

    # Ensure incidentType matches expected API format
    category = category.replace("_", " ")  # Convert underscores to spaces

    payload = {
        "vehicleIncidentDetails": [
            {
                "vehicleNumber": vehicle_number,
                "cameraLocation": camera_location,
                "chainage": chainage,
                "category": category,
                "incidentType": incident_type,
                "confidenceValue": confidence, # Confidence value for the incident
                "description": description,
                "videoUrl": ",".join(video_urls) if video_urls else "",
                "imagesUrl": ",".join(images_urls) if images_urls else "",
                "cameraId": camera_id,
                "cameraIp": camera_ip,
                "timestamp": timestamp
            }
        ]
    }

    # *Print Payload Before Sending*
    print(f"📡 Camera {camera_id} - Final Incident Payload:")
    print(json.dumps(payload, indent=4))

    try:
        response = requests.post(INCIDENT_URL, headers=headers, json=payload)
        print(f"📨 Camera {camera_id} - Server Response: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error sending incident data for Camera {camera_id}: {e}")


# def handle_incident(vehicle_number, camera_location, chainage, category, incident_type, 
#                    description, camera_id, rtsp, timestamp, evidence_path):
#     """Handle incident with image evidence only"""
#     try:
#         print(f"Handling incident for Camera {camera_id}: {incident_type} at {camera_location}")
        
#         # Ensure all parameters are strings (with defaults)
#         params = {
#             "vehicle_number": str(vehicle_number or ""),
#             "camera_location": str(camera_location or "Unknown"),
#             "chainage": str(chainage or "Unknown"),
#             "category": str(category or "Unknown"),
#             "incident_type": str(incident_type or "Unknown"),
#             "description": str(description or "No description"),
#             "camera_id": str(camera_id or "0"),
#             "rtsp": str(rtsp or ""),
#             "timestamp": str(timestamp or "")
#         }
        
#         # Make sure evidence_path is a list
#         if evidence_path:
#             if isinstance(evidence_path, str):
#                 evidence_paths = [evidence_path]
#             else:
#                 evidence_paths = evidence_path
#         else:
#             evidence_paths = []
        
#         # Convert evidence images to HTTP URLs
#         images_urls = convert_to_http_files(evidence_paths) if evidence_paths else []
#         if not images_urls:
#             print(f"⚠️ Camera {camera_id}: No valid evidence uploaded. Skipping incident report.")
#             return None
            
#         send_vehicle_incident_data(
#             **params,
#             video_urls=[],    # No video evidence is used
#             images_urls=images_urls
#         )
        
#     except Exception as e:
#         print(f"Error handling incident: {str(e)}")
#     finally:
#         # Cleanup evidences after upload
#         for ep in evidence_paths:
#             if ep and os.path.exists(ep):
#                 try:
#                     os.remove(ep)
#                 except Exception as e:
#                     print(f"Error removing evidence file: {str(e)}")



def handle_incident(vehicle_number, camera_location, chainage, category, incident_type, 
                    description,confidence, camera_id, rtsp, timestamp, image_evidence_paths, video_evidence_path):
    """Handle incident and send both image and video evidences"""
    try:
        print(f"Handling incident for Camera {camera_id}: {incident_type} at {camera_location}")
        
        # Convert evidences to HTTP URLs (assuming convert_to_http_files supports lists)
        images_urls = convert_to_http_files(image_evidence_paths) if image_evidence_paths else []
        video_urls = convert_to_http_files([video_evidence_path]) if video_evidence_path else []
        
        if not images_urls and not video_urls:
            print(f"⚠️ Camera {camera_id}: No valid evidence uploaded. Skipping incident report.")
            return None
            
        send_vehicle_incident_data(
            vehicle_number=str(vehicle_number or ""),
            camera_location=str(camera_location or "Unknown"),
            chainage=str(chainage or "Unknown"),
            category=str(category or "Unknown"),
            incident_type=str(incident_type or "Unknown"),
            description=str(description or "No description"),
            confidence=str(confidence or "7.2"),
            camera_id=str(camera_id or "0"),
            rtsp=str(rtsp or ""),
            timestamp=str(timestamp or ""),
            video_urls=video_urls,
            images_urls=images_urls
        )
        
    except Exception as e:
        print(f"Error handling incident: {str(e)}")
    finally:
        # Cleanup: remove the local evidence files
        for file_path in image_evidence_paths or []:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing image evidence file: {str(e)}")
        if video_evidence_path and os.path.exists(video_evidence_path):
            try:
                os.remove(video_evidence_path)
            except Exception as e:
                print(f"Error removing video evidence file: {str(e)}")