#main.py
import cv2
import threading
import torch
import numpy as np
import time
import gc
import functools
from datetime import datetime, timezone
import pytz
import functools
from collections import deque
from ultralytics import YOLO
from save_to_server1 import *
import concurrent.futures
import math
from deep_sort_realtime.deepsort_tracker import DeepSort

# === SYSTEM INFORMATION ===
def get_ist_time():
    """Convert UTC to IST"""
    utc_time = datetime.now(timezone.utc)
    ist = pytz.timezone('Asia/Kolkata')
    return utc_time.astimezone(ist)

def format_ist_time():
    """Format current IST time as string"""
    return get_ist_time().strftime('%Y-%m-%d %H:%M:%S')

SYSTEM_INFO = {
    "current_user": "Watsoo Exp. Pvt. Ltd.",
    "system_name": "Watsoo AI System",
    "timezone": "IST",
    "start_time": format_ist_time(),
    "version": "4.0.0"
}

# === CONFIGURATION ===
SOURCE_TYPE = "mp4"  # or "rtsp"
FRAME_SKIP = 25
CONF_THRESH = 0.45
CAMERA_RESOLUTION = (640, 640)  # NEED TO BE DIVISIBLE BY 32
MAX_QUEUE_SIZE = 3
BATCH_SIZE = 4
CLEANUP_INTERVAL = 180
COOLDOWN_PERIOD = 60

# === VEHICLE DETECTION CONFIG ===
STATIONARY_FRAME_THRESHOLD = 30
STATIONARY_VELOCITY_THRESHOLD = 0.5
IOU_THRESHOLD = 0.72
CONGESTION_THRESHOLD = 5

# === GPU OPTIMIZATION ===
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_num_threads(2)

# print(f"Using device: {device}")
# print(f"System initialized at {SYSTEM_INFO['start_time']} UTC")
# print(f"Current user: {SYSTEM_INFO['current_user']}")

# === CLASS MAPPINGS ===
CUSTOM_CLASSES = {0: "puddle", 
                  1: "accident", 
                  2: "fire", 3: "rocks", 4: "smoke"}
COCO_CLASSES = {0: "person", 1: "bicycle", 3: "motorcycle", 19: "cow",
                17: "horse", 20: "elephant", 21: "bear"}
VEHICLE_CLASSES = {2: "car", 7: "truck"}
                #    5: "bus", }

# Custom confidence thresholds for selected classes
CUSTOM_CONF_THRESHOLDS = {
    "fire": 0.72,
    "puddle": 0.81,
    "rocks": 0.89,
    "smoke": 0.4,
    "accident": 0.5,
    "person":0.72,
    "bicycle":0.81,
    "motorcycle":0.72,
    "horse":0.72,
    "elephant":0.81,
    "bear":0.81,
    # "car":0.72,
    "cow":0.81,
    "truck":0.81,
}


# === INCIDENT TYPE MAPPING ===
INCIDENT_TYPE_MAP = {
    "Accident_Detection": ["accident"],
    "Pedestrian_On_Road": ["person"],
    "Animals_Detected": ["elephant", "cow", "horse"],
    "Object_On_Road": ["bicycle", "motorcycle", "puddle", "rocks"],
    "Smoke_Or_Fire_Detection": ["fire", "smoke"],
    "Stopped_Vehicle": ["is_stationary"],
    "Traffic_Congestion": ["congestion"],
    "Wrong_Way_Driving": ["wrong_way_vehicle"],
    "Weather_Condition_Alert": "Low_Visibility",
}

# === TRACKING STATE ===
last_sent = {}

class RTSPStream:
    def __init__(self, url, camera_id=None, frame_skip=FRAME_SKIP, resolution=CAMERA_RESOLUTION, target_fps=25):
        # Core attributes
        self.stream = None
        self.running = False
        self.thread = None
        # Increase the buffer size to hold enough frames (here, 30 frames roughly for 2s at ~15 fps)
        self.frame_buffer = deque(maxlen=60)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2
        
        if camera_id is None:
            raise ValueError("camera_id is required")
        
        # Stream settings
        self.url = url
        self.camera_id = camera_id
        self.frame_skip = frame_skip
        self.resolution = (
            resolution[0] - (resolution[0] % 32),
            resolution[1] - (resolution[1] % 32)
        )
        
        # FPS control
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0
        
        # Start stream thread
        self.running = True
        self.thread = threading.Thread(
            target=self._update,
            name=f"Stream-{camera_id}",
            daemon=True
        )
        self._log("Initializing stream...")
        self.thread.start()
    
    def _log(self, message):
        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] [Camera: {self.camera_id}] {message}")
    
    def _connect(self):
        """Initialize video capture with optimized settings."""
        try:
            if self.stream is not None:
                self.stream.release()
            
            # For MP4 sources, ensure file exists
            if not self.url.startswith("rtsp://"):
                if not os.path.exists(self.url):
                    self._log(f"Error: File not found: {self.url}")
                    return False
            
            # Create capture with appropriate backend
            if self.url.startswith("rtsp://"):
                self.stream = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            else:
                self.stream = cv2.VideoCapture(self.url)
            
            if not self.stream.isOpened():
                self._log(f"Failed to open stream: {self.url}")
                return False
            
            # Set capture properties
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.stream.set(cv2.CAP_PROP_FPS, min(self.target_fps, self.frame_skip))
            
            # Read a test frame to verify connection
            ret, test_frame = self.stream.read()
            if not ret or test_frame is None:
                self._log("Failed to read initial frame")
                return False
            
            self._log("Stream connected successfully")
            self.reconnect_attempts = 0
            return True
        
        except Exception as e:
            self._log(f"Connection error: {str(e)}")
            return False
            
    def _update(self):
        """Frame capture loop with error handling and reconnection logic."""
        retry_count = 0
        while self.running:
            try:
                if self.stream is None:
                    if not self._connect():
                        retry_count += 1
                        self._log(f"Attempting reconnection ({retry_count}/{self.max_reconnect_attempts})")
                        if retry_count >= self.max_reconnect_attempts:
                            self._log("Max reconnection attempts reached")
                            self.running = False
                            break
                        time.sleep(self.reconnect_delay)
                        continue
                    retry_count = 0
                
                if not self.stream.isOpened():
                    raise Exception("Stream disconnected")
                
                ret, frame = self.stream.read()
                if not ret or frame is None:
                    raise Exception("Failed to read frame")
                
                if frame.shape[:2] != self.resolution[::-1]:
                    frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LINEAR)
                
                # Append the current frame with a timestamp to the buffer
                self.frame_buffer.append((frame.copy(), datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')))
                
                # FPS control
                current_time = time.time()
                sleep_duration = self.frame_interval - (current_time - self.last_frame_time)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                self.last_frame_time = time.time()
                
            except Exception as e:
                self._log(f"Stream error: {str(e)}")
                if self.stream:
                    self.stream.release()
                self.stream = None
                time.sleep(self.reconnect_delay)
    
    def get_recent_frames(self, duration=4, fps=15):
        """Return the most recent frames (only image data) from the buffer for the specified duration."""
        required_frames = int(duration * fps)
        frames = [frame for frame, ts in list(self.frame_buffer)[-required_frames:]]
        return frames
    
    def read(self):
        """Thread-safe read: return last frame (with timestamp) from the buffer."""
        try:
            if not self.running:
                return False, None, None
            if len(self.frame_buffer) > 0:
                return True, *self.frame_buffer[-1]
            return False, None, None
        except Exception:
            return False, None, None
    
    def stop(self):
        """Clean up resources and stop stream."""
        self.running = False
        if self.stream is not None:
            self.stream.release()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._log("Stream stopped")
    
    def is_alive(self):
        return self.running and self.thread.is_alive()
        
    def __del__(self):
        if self.running:
            self.stop()

class VehicleTracker:
    """Efficient vehicle tracking with optimized calculations and direction detection"""
    def __init__(self):
        self.tracked_vehicles = {}
        self.vehicle_history = {}
        self.last_cleanup = time.time()
        self.vehicle_directions = {}
        self.kalman_filters = {}  # Store Kalman filters for each vehicle
        self.position_history = {}  # Store recent positions for spatial variance

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def calculate_iou(box1, box2):
        """Cached IOU calculation"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def calculate_direction(self, current_box, previous_box):
        """Calculate movement direction vector between two bounding boxes."""
        # Get center points of current and previous boxes
        current_center_x = (current_box[0] + current_box[2]) / 2
        current_center_y = (current_box[1] + current_box[3]) / 2
        
        previous_center_x = (previous_box[0] + previous_box[2]) / 2
        previous_center_y = (previous_box[1] + previous_box[3]) / 2
        
        # Calculate direction vector (delta x, delta y)
        dx = current_center_x - previous_center_x
        dy = current_center_y - previous_center_y
        
        # Return movement vector
        return (dx, dy)
    def initialize_kalman_filter(self, initial_position):
        """Initialize a Kalman filter for a vehicle."""
        kf = cv2.KalmanFilter(6, 4)  # 6D state (x, y, vx, vy, ax, ay), 4D measurement (x, y, w, h)
        kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                        [0, 1, 0, 1, 0, 0.5],
                                        [0, 0, 1, 0, 1, 0],
                                        [0, 0, 0, 1, 0, 1],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.05  # Adjust noise
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1  # Adjust noise
        kf.statePost = np.array(
            [[initial_position[0]], [initial_position[1]], [0], [0], [0], [0]],
            np.float32,
        )
        return kf

    def is_wrong_way(self, camera_id, vehicle_id, lane_directions):
        """Determine if a vehicle is going wrong way by computing the angle between vectors."""
        if camera_id not in self.vehicle_directions or vehicle_id not in self.vehicle_directions[camera_id]:
            return False
            
        # Get vehicle's movement vector and current bounding box/position
        direction_vector, position = self.vehicle_directions[camera_id][vehicle_id]
        v_dx, v_dy = direction_vector

        # Calculate magnitude (speed)
        v_magnitude = math.hypot(v_dx, v_dy)
        # Check if there's enough movement
        if v_magnitude < 3:
            return False

        # Normalize vehicle's movement vector
        norm_vehicle = (v_dx / v_magnitude, v_dy / v_magnitude)

        # For each lane, check if vehicle is in region and compare direction
        for lane_id, lane_info in lane_directions.items():
            lane_region = lane_info["region"]
            expected_direction = lane_info["direction"]
            
            # Compute vehicle center
            x_center = (position[0] + position[2]) / 2
            y_center = (position[1] + position[3]) / 2
            
            if (lane_region[0] <= x_center <= lane_region[2] and
                lane_region[1] <= y_center <= lane_region[3]):
                
                # Normalize expected direction vector
                e_dx, e_dy = expected_direction
                e_magnitude = math.hypot(e_dx, e_dy)
                if e_magnitude == 0:
                    continue
                norm_expected = (e_dx / e_magnitude, e_dy / e_magnitude)
                
                # Compute dot product
                cos_theta = norm_vehicle[0] * norm_expected[0] + norm_vehicle[1] * norm_expected[1]
                # Clamp cosine value to avoid floating point errors
                cos_theta = max(min(cos_theta, 1.0), -1.0)
                # Calculate the angle in degrees
                angle = math.degrees(math.acos(cos_theta))
                
                # If the angle between movement direction and expected direction exceeds 90°, it is wrong-way
                if angle > 90:
                    return True
                    
        return False

    # def update_tracks(self, camera_id, current_vehicles, frame_count):
    #     """Update vehicle tracks using Kalman filtering, adaptive velocity, and a stationary flag."""
    #     if camera_id not in self.tracked_vehicles:
    #         self.tracked_vehicles[camera_id] = {}
    #     if camera_id not in self.vehicle_directions:
    #         self.vehicle_directions[camera_id] = {}
    #     if camera_id not in self.kalman_filters:
    #         self.kalman_filters[camera_id] = {}
    #     if camera_id not in self.position_history:
    #         self.position_history[camera_id] = {}

    #     # Convert legacy 3-tuple tracking data to 4-tuple by adding is_stationary flag
    #     for vehicle_id, data in self.tracked_vehicles[camera_id].items():
    #         if len(data) == 3:
    #             self.tracked_vehicles[camera_id][vehicle_id] = (data[0], data[1], data[2], False)

    #     new_tracked_vehicles = {}
    #     stationary_count = 0

    #     for current_box in current_vehicles:
    #         x1, y1, x2, y2 = current_box
    #         centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
    #         width = x2 - x1
    #         height = y2 - y1
    #         measurement = np.array([[centroid[0]], [centroid[1]], [width], [height]], np.float32)

    #         best_match = None
    #         best_iou = IOU_THRESHOLD

    #         for vehicle_id, (tracked_box, first_seen, consecutive_frames, is_stationary) in self.tracked_vehicles[
    #             camera_id].items():
    #             iou = self.calculate_iou(current_box, tracked_box)
    #             if iou > best_iou:
    #                 best_iou = iou
    #                 best_match = vehicle_id

    #         if best_match:
    #             tracked_box, first_seen, consecutive_frames, is_stationary = self.tracked_vehicles[camera_id][
    #                 best_match]
    #             consecutive_frames += 1

    #             # Kalman filter update
    #             kf = self.kalman_filters[camera_id][best_match]
    #             kf.correct(measurement)
    #             predicted_state = kf.predict()
    #             predicted_centroid = (predicted_state[0, 0], predicted_state[1, 0])
    #             predicted_width = predicted_state[2, 0]
    #             predicted_height = predicted_state[3, 0]
    #             predicted_box = (
    #                 int(predicted_centroid[0] - predicted_width / 2),
    #                 int(predicted_centroid[1] - predicted_height / 2),
    #                 int(predicted_centroid[0] + predicted_width / 2),
    #                 int(predicted_centroid[1] + predicted_height / 2),
    #             )

    #             # Velocity calculation from Kalman filter
    #             velocity = math.hypot(predicted_state[2, 0], predicted_state[3, 0])

    #             # Adaptive velocity threshold
    #             vehicle_area = width * height
    #             adaptive_velocity_threshold = STATIONARY_VELOCITY_THRESHOLD * (
    #                 0.5 + vehicle_area / CAMERA_RESOLUTION[0] / CAMERA_RESOLUTION[1]
    #             )

    #             # Update direction with Kalman prediction
    #             direction_vector = self.calculate_direction(predicted_box, tracked_box)
    #             if best_match in self.vehicle_directions[camera_id]:
    #                 prev_direction, prev_position = self.vehicle_directions[camera_id][best_match]
    #                 smoothed_dx = 0.7 * direction_vector[0] + 0.3 * prev_direction[0]
    #                 smoothed_dy = 0.7 * direction_vector[1] + 0.3 * prev_direction[1]
    #                 direction_vector = (smoothed_dx, smoothed_dy)
    #             self.vehicle_directions[camera_id][best_match] = (direction_vector, predicted_box)

    #             # Stationary determination logic with the flag
    #             was_stationary = is_stationary
    #             is_stationary = velocity < adaptive_velocity_threshold

    #             if is_stationary:
    #                 stationary_count += 1

    #             new_tracked_vehicles[best_match] = (predicted_box, first_seen, consecutive_frames, is_stationary)

    #         else:
    #             vehicle_id = f"vehicle_{frame_count}_{len(new_tracked_vehicles)}"
    #             new_tracked_vehicles[vehicle_id] = (current_box, frame_count, 1, False)  # Initialize is_stationary to False
    #             self.vehicle_directions[camera_id][vehicle_id] = ((0, 0), current_box)
    #             self.kalman_filters[camera_id][
    #                 vehicle_id] = self.initialize_kalman_filter((centroid[0], centroid[1], width, height))
    #             self.position_history[camera_id][vehicle_id] = deque(maxlen=10)
    #             self.position_history[camera_id][vehicle_id].append(centroid)

    #     # Cleanup
    #     current_time = time.time()
    #     if current_time - self.last_cleanup > CLEANUP_INTERVAL:
    #         self.last_cleanup = current_time
    #         new_tracked_vehicles = {
    #             vid: data for vid, data in new_tracked_vehicles.items()
    #             if frame_count - data[1] < CLEANUP_INTERVAL
    #         }
    #         if camera_id in self.vehicle_directions:
    #             self.vehicle_directions[camera_id] = {
    #                 vid: data for vid, data in self.vehicle_directions[camera_id].items()
    #                 if vid in new_tracked_vehicles
    #             }
    #         if camera_id in self.kalman_filters:
    #             self.kalman_filters[camera_id] = {
    #                 vid: kf for vid, kf in self.kalman_filters[camera_id].items()
    #                 if vid in new_tracked_vehicles
    #             }
    #         if camera_id in self.position_history:
    #             self.position_history[camera_id] = {
    #                 vid: history for vid, history in self.position_history[camera_id].items()
    #                 if vid in new_tracked_vehicles
    #             }

    #     self.tracked_vehicles[camera_id] = new_tracked_vehicles
    #     return stationary_count
    
    def update_tracks(self, camera_id, current_vehicles, frame_count):
        """Improved method for detecting stalled vehicles with speed & IoU filtering."""
        if camera_id not in self.tracked_vehicles:
            self.tracked_vehicles[camera_id] = {}

        new_tracked_vehicles = {}
        stationary_count = 0

        for current_box in current_vehicles:
            x1, y1, x2, y2 = current_box
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            width, height = x2 - x1, y2 - y1
            area = width * height

            best_match = None
            best_iou = IOU_THRESHOLD

            for vehicle_id, (tracked_box, first_seen, consecutive_frames) in self.tracked_vehicles[camera_id].items():
                iou = self.calculate_iou(current_box, tracked_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = vehicle_id

            if best_match:
                tracked_box, first_seen, consecutive_frames = self.tracked_vehicles[camera_id][best_match]
                consecutive_frames += 1

                # Calculate speed (displacement between frames)
                prev_x1, prev_y1, prev_x2, prev_y2 = tracked_box
                prev_centroid = ((prev_x1 + prev_x2) / 2, (prev_y1 + prev_y2) / 2)
                speed = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))

                # Adaptive threshold based on vehicle size
                adaptive_velocity_threshold = max(1.0, STATIONARY_VELOCITY_THRESHOLD * (area / 50000))

                # Stationary determination: IoU + Speed Check + Time Filter
                is_stationary = (iou > 0.85 and speed < adaptive_velocity_threshold and consecutive_frames > STATIONARY_FRAME_THRESHOLD)

                if is_stationary:
                    stationary_count += 1
                    if best_match not in last_sent[camera_id] or (time.time() - last_sent[camera_id][best_match]) > COOLDOWN_PERIOD:
                        last_sent[camera_id][best_match] = time.time()
                        print(f"🚨 STOPPED VEHICLE DETECTED: {best_match}, Speed: {speed:.2f}")

                new_tracked_vehicles[best_match] = (current_box, first_seen, consecutive_frames)

            else:
                vehicle_id = f"vehicle_{frame_count}_{len(new_tracked_vehicles)}"
                new_tracked_vehicles[vehicle_id] = (current_box, frame_count, 1)

        self.tracked_vehicles[camera_id] = new_tracked_vehicles
        return stationary_count

        
    def detect_wrong_way_vehicles(self, camera_id, lane_directions):
        """
        Detect vehicles moving in the opposite direction of expected lane flow.
        Returns a list of (vehicle_id, position) tuples for vehicles going the wrong way.
        """
        wrong_way_vehicles = []
        
        if camera_id not in self.tracked_vehicles or camera_id not in self.vehicle_directions:
            return wrong_way_vehicles
            
        for vehicle_id, (tracked_box, first_seen, consecutive_frames) in self.tracked_vehicles[camera_id].items():
            # Get vehicle's movement vector and current position
            if vehicle_id not in self.vehicle_directions[camera_id]:
                continue
                
            direction_vector, position = self.vehicle_directions[camera_id][vehicle_id]
            vehicle_dx, vehicle_dy = direction_vector
            
            # Find which lane the vehicle is in
            for lane_id, lane_info in lane_directions.items():
                lane_region = lane_info["region"]
                expected_direction = lane_info["direction"]
                
                # Calculate vehicle center
                x_center = (position[0] + position[2]) / 2
                y_center = (position[1] + position[3]) / 2
                
                # Check if vehicle is in this lane
                if (lane_region[0] <= x_center <= lane_region[2] and 
                    lane_region[1] <= y_center <= lane_region[3]):
                    
                    expected_dx, expected_dy = expected_direction
                    # Calculate dot product to check direction
                    dot_product = vehicle_dx * expected_dx + vehicle_dy * expected_dy
                    
                    # Vehicle is going against traffic if dot product is negative 
                    # and movement magnitude is significant
                    if dot_product < 0 and (abs(vehicle_dx) > 3 or abs(vehicle_dy) > 3):
                        wrong_way_vehicles.append((vehicle_id, position))
                    break
                    
        return wrong_way_vehicles

class PerformanceMonitor:
    """Monitor and log system performance"""
    def __init__(self):
        self.start_time = get_ist_time()
        self.frame_counts = {}
        self.processing_times = []
        self.last_log = time.time()
        
    def update_stats(self, camera_id, processing_time):
        """Update performance statistics"""
        if camera_id not in self.frame_counts:
            self.frame_counts[camera_id] = 0
        self.frame_counts[camera_id] += 1
        self.processing_times.append(processing_time)
        
        # Log performance periodically
        current_time = time.time()
        if current_time - self.last_log > 60:  # Log every minute
            self.log_performance()
            self.last_log = current_time
    
    def log_performance(self):
        """Log current performance metrics"""
        stats = self.get_stats()
        print("\n=== Performance Statistics ===")
        print(f"Time: {format_ist_time()} IST")
        print(f"User: {SYSTEM_INFO['current_user']}")
        print(f"FPS: {stats['fps']:.2f}")
        print(f"Average processing time: {stats['avg_processing_time']*1000:.2f}ms")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"System uptime: {stats['uptime']:.2f}s")
        print("===========================\n")
    
    def get_stats(self):
        """Get current performance statistics"""
        avg_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        return {
            "fps": fps,
            "avg_processing_time": avg_time,
            "total_frames": sum(self.frame_counts.values()),
            "uptime": (get_ist_time() - self.start_time).total_seconds(),
            "user": SYSTEM_INFO['current_user']
        }

def load_models():
    """Load and optimize models"""
    print(f"Loading models on {device}...")
    model_coco = YOLO("/home/kaustav/AIML/HAZARD/yolov8s.onnx", task="detect")
    model_custom = YOLO("/home/kaustav/AIML/HAZARD/FINAL WEIGHTS/kaggle_weights_MARCH/kaggle/working/runs/detect/train/weights/best.onnx", task="detect")
    print("Models loaded successfully")
    return model_coco, model_custom

def get_incident_type(label):
    """Map detected label to incident type"""
    for incident_type, labels in INCIDENT_TYPE_MAP.items():
        if label in labels:
            return incident_type
    return "Unknown"


def draw_lane_regions(frame, camera_config):
    """Draw lane regions and direction arrows"""
    if "lane_directions" not in camera_config:
        return frame
        
    for lane_id, lane_info in camera_config["lane_directions"].items():
        region = lane_info["region"]
        direction = lane_info["direction"]
        
        # DRAW LANE REGION
        # cv2.rectangle(frame, (region[0], region[1]), (region[2], region[3]), (0, 255, 255), 1)
        
        # Draw direction arrow
        center_x = (region[0] + region[2]) // 2
        center_y = (region[1] + region[3]) // 2
        end_x = center_x + direction[0] * 30
        end_y = center_y + direction[1] * 30
        
        cv2.arrowedLine(frame, (center_x, center_y), (int(end_x), int(end_y)), (0, 255, 255), 2)
        cv2.putText(frame, lane_id, (region[0] + 5, region[1] + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame

def define_lane_directions(camera_width, camera_height, num_lanes=2, horizontal=True):
    """
    Automatically define lane regions and directions based on camera resolution
    
    Args:
        camera_width: Width of camera frame in pixels
        camera_height: Height of camera frame in pixels
        num_lanes: Number of lanes the frame should be divided into
        horizontal: If True, lanes are arranged horizontally (dividing vertically); 
                    else, lanes are vertical (dividing horizontally)
                    
    Returns:
        A dictionary with lane regions and expected direction vectors.
    """
    lane_directions = {}
    
    if horizontal:
        # For horizontal lanes, divide the height equally.
        lane_height = camera_height // num_lanes
        for i in range(num_lanes):
            lane_id = f"lane{i+1}"
            y1 = i * lane_height
            y2 = (i + 1) * lane_height
            # Example: alternate expected directions (adjust as needed or set manually)
            direction = (1, 0) if i % 2 == 0 else (-1, 0)
            lane_directions[lane_id] = {
                "region": (0, y1, camera_width, y2),
                "direction": direction
            }
    else:
        # For vertical lanes, divide the width equally.
        lane_width = camera_width // num_lanes
        for i in range(num_lanes):
            lane_id = f"lane{i+1}"
            x1 = i * lane_width
            x2 = (i + 1) * lane_width
            # Example: alternate expected directions
            direction = (0, 1) if i % 2 == 0 else (0, -1)
            lane_directions[lane_id] = {
                "region": (x1, 0, x2, camera_height),
                "direction": direction
            }
    
    return lane_directions
# def auto_detect_lane_regions(frame, num_lanes=2):
#     """
#     Automatically detect lane regions in the frame using edge detection and Hough transform.
    
#     Args:
#         frame: Input BGR image.
#         num_lanes: Number of lanes to detect.
    
#     Returns:
#         A dictionary with lane regions and estimated directions.
#     """
#     import numpy as np  # ensure numpy is imported
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)

#     lane_regions = {}
#     height, width = frame.shape[:2]

#     if lines is not None:
#         # For simplicity, assume lanes arranged vertically by clustering x coordinates.
#         x_coords = [ (line[0][0] + line[0][2]) // 2 for line in lines ]
#         x_coords.sort()
#         group_size = max(1, len(x_coords) // num_lanes)
#         for i in range(num_lanes):
#             group = x_coords[i*group_size:(i+1)*group_size]
#             if not group:
#                 continue
#             median_x = int(np.median(group))
#             # Define region based on a window around the median
#             region_width = width // (num_lanes + 1)
#             lane_regions[f"lane{i+1}"] = {
#                 "region": (max(0, median_x - region_width//2), 0,
#                            min(width, median_x + region_width//2), height),
#                 "direction": (0, 1)  # DIRECTION CAN BE ADJUSTED AS NEEDED
#                 # "direction": (0, -1)
#                 # "direction": (1, 0)
#                 # "direction": (-1, 0)
#             }
#     else:
#         # Fallback to equally divided vertical lanes
#         lane_regions = define_lane_directions(width, height, num_lanes=num_lanes, horizontal=False)

#     return lane_regions
def auto_detect_lane_regions(frame, num_lanes=2):
    """
    Automatically detect lane regions using edge detection, Hough transform,
    and then fit a polynomial curve to each lane (left & right halves) to account for curvature.
    """
    import numpy as np  # ensure numpy is imported
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
    lane_regions = {}
    height, width = frame.shape[:2]
    
    if lines is not None:
        # Separate lines into left and right groups using the center x coordinate.
        centers = np.array([ (line[0][0] + line[0][2]) / 2.0 for line in lines ])
        left_indices = np.where(centers < width / 2)[0]
        right_indices = np.where(centers >= width / 2)[0]
        
        offset = 50  # adjust region width around the fitted curve
        
        # Fit polynomial to left lane lines if available
        if len(left_indices) > 0:
            left_points = []
            for idx in left_indices:
                x1, y1, x2, y2 = lines[idx][0]
                left_points.extend([(x1, y1), (x2, y2)])
            left_points = np.array(left_points)
            poly_left = np.polyfit(left_points[:,1], left_points[:,0], deg=2)
            x_left_top = int(np.polyval(poly_left, 0))
            x_left_bottom = int(np.polyval(poly_left, height))
            lane_regions["lane1"] = {
                "region": (max(0, x_left_top - offset), 0, min(width, x_left_bottom + offset), height),
                "direction": (0, 1)
            }
        
        # Fit polynomial to right lane lines if available
        if len(right_indices) > 0:
            right_points = []
            for idx in right_indices:
                x1, y1, x2, y2 = lines[idx][0]
                right_points.extend([(x1, y1), (x2, y2)])
            right_points = np.array(right_points)
            poly_right = np.polyfit(right_points[:,1], right_points[:,0], deg=2)
            x_right_top = int(np.polyval(poly_right, 0))
            x_right_bottom = int(np.polyval(poly_right, height))
            lane_regions["lane2"] = {
                "region": (max(0, x_right_top - offset), 0, min(width, x_right_bottom + offset), height),
                "direction": (0, 1)
            }
            
    else:
        # Fallback to equally divided vertical lanes if no lines detected
        lane_regions = define_lane_directions(width, height, num_lanes=num_lanes, horizontal=False)
    
    return lane_regions
 
# DEBUG_PRINT_WEATHER = True  # Global flag; set to False to disable weather condition debug prints

# def detect_weather_condition(frame, brightness_threshold=40, contrast_threshold=45, night_threshold=20):
#     """
#     Detect weather conditions (e.g. fog, smog, rain) and low visibility due to night.
    
#     If the average brightness is extremely low (below night_threshold), returns a 'night' condition.
#     Otherwise, if contrast falls under contrast_threshold, distinguishes among fog, smog, and rain based on avg brightness.
    
#     Returns:
#         A tuple (is_low_visibility, condition_type, severity)
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     avg_brightness = np.mean(gray)
#     contrast = np.std(gray)

#     # Global debug print
#     if DEBUG_PRINT_WEATHER:
#         print(f"[DEBUG] detect_weather_condition - avg_brightness: {avg_brightness:.2f}, contrast: {contrast:.2f}")
#         print(f"[DEBUG] Thresholds - brightness_threshold: {brightness_threshold}, contrast_threshold: {contrast_threshold}, night_threshold: {night_threshold}")

#     # Detect nighttime based solely on extremely low brightness.
#     if avg_brightness < night_threshold:
#         return True, "night", "severe"
    
#     # If contrast is low, further distinguish known conditions.
#     if contrast < contrast_threshold:
#         if avg_brightness > 117:
#             return True, "fog", "moderate"
#         elif avg_brightness < brightness_threshold:
#             return True, "smog", "severe"
#         else:
#             return True, "Unknown", "moderate"
    
#     return False, "clear", "none"
DEBUG_PRINT_WEATHER = True  # Set to False in production

def detect_weather_condition(frame, brightness_threshold=40, contrast_threshold=45, night_threshold=20):
    """
    Enhanced weather condition detection using multiple image metrics.
    
    Metrics used:
      1. Global brightness and contrast
      2. Edge density (via Canny edge detection)
      3. Image entropy (to capture detail richness)
    
    Conditions:
      - If brightness < night_threshold: "night" with severe severity.
      - If contrast is low (below contrast_threshold):
            • If brightness is high (> 117) and edge density is very low, it's "fog" (moderate).
            • If brightness is low (< brightness_threshold) and entropy is very low, it's "smog" (severe).
            • Otherwise, if contrast is low, it’s likely "rain" (moderate).
      - If none of these conditions hold, visibility is considered "clear".
      
    Returns:
        Tuple (is_low_visibility: bool, condition_type: str, severity: str)
    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Compute edge density using Canny; fewer edges suggest low detail (fog/smog)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0
    
    # Compute image entropy to estimate image information; lower entropy indicates loss of details
    histogram = cv2.calcHist([gray], [0], None, [256], [0,256])
    histogram_norm = histogram.ravel() / histogram.sum()
    entropy = -np.sum(histogram_norm * np.log2(histogram_norm + 1e-7))
    
    # Debug prints
    if DEBUG_PRINT_WEATHER:
        print("\n[DEBUG] Enhanced Weather Analysis:")
        print(f"  → Brightness: {avg_brightness:.2f} (night threshold: {night_threshold})")
        print(f"  → Contrast: {contrast:.2f} (low contrast threshold: {contrast_threshold})")
        print(f"  → Edge Density: {edge_density:.3f}")
        print(f"  → Entropy: {entropy:.2f}")
        print(f"  → Brightness low threshold: {brightness_threshold}")
    
    # Check for nighttime: extremely low brightness indicates night
    if avg_brightness < night_threshold:
        return True, "night", "severe"
    
    # If contrast is low, assess further using edge density and entropy
    if contrast < contrast_threshold:
        # Assume that high brightness but low detail typically indicates fog
        if avg_brightness > 117 and edge_density < 0.09 or entropy < 7.0:
            return True, "fog", "moderate"
        # Assume that very low brightness together with low entropy (loss of texture) indicates smog
        elif avg_brightness < brightness_threshold and entropy < 6.5:
            return True, "smog", "severe"
        # Otherwise, if details are moderately suppressed, we assume rain conditions
        else:
            return True, "Unknown", "moderate"
    
    return False, "clear", "none"

def enhance_frame(frame, gamma=1.5):
    """
    Enhance low-light images using gamma correction and CLAHE.
    Adjust gamma value adaptively if needed.
    """
    # Gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    gamma_corrected = cv2.LUT(frame, table)
    
    # Convert to LAB and apply CLAHE to the L channel
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def batch_process_frames(models, frames, camera_configs, vehicle_tracker, perf_monitor):
    """Process multiple frames in a batch"""
    if not frames:
        return []

    model_coco, model_custom = models
    start_time = time.time()

    # Prepare frames with correct format (BCHW)
    batch_tensor = []
    # In batch_process_frames(), before converting to tensor:
    for frame, _ in frames:
        enhanced_frame = enhance_frame(frame)  # enhance low-light images
        frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        frame_chw = np.transpose(frame_rgb, (2, 0, 1))
        frame_tensor = torch.from_numpy(frame_chw).float() / 255.0
        batch_tensor.append(frame_tensor)

    # Stack tensors and ensure correct shape
    batch_tensor = torch.stack(batch_tensor)  # Don't move to device explicitly

    # Process in single batch using predict mode with device argument
    results_coco = model_coco.predict(source=batch_tensor, device=device)
    results_custom = model_custom.predict(source=batch_tensor, device=device)
    
    all_detections = []

    for i in range(len(frames)):
        frame, timestamp = frames[i]
        camera_config = camera_configs[i % len(camera_configs)]
        camera_id = camera_config["camera_id"]
        current_time = time.time()

        if camera_id not in last_sent:
            last_sent[camera_id] = {}

        frame_detections = []
        current_vehicles = []
        
        # Process COCO detections
        for box in results_coco[i].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            # Determine label from the known mappings
            label = None
            if cls in COCO_CLASSES:
                label = COCO_CLASSES[cls]
            elif cls in VEHICLE_CLASSES:
                label = VEHICLE_CLASSES[cls]

            # Skip if no label found
            if not label:
                continue

            # Use custom threshold if configured; otherwise, use CONF_THRESH
            threshold = CUSTOM_CONF_THRESHOLDS.get(label, CONF_THRESH)
            # Print the detection confidence and the threshold being used
            print(f"COCO Detection: label={label}, conf={conf:.2f}, threshold={threshold:.2f}")
            if conf < threshold:
                continue

            # For vehicle boxes, add to tracking for later use
            if cls in VEHICLE_CLASSES:
                current_vehicles.append((x1, y1, x2, y2))
                continue
            current_time = time.time()
            if label in last_sent[camera_id] and (current_time - last_sent[camera_id][label]) < COOLDOWN_PERIOD:
                continue

            last_sent[camera_id][label] = current_time
            incident_type = get_incident_type(label)
            frame_detections.append({
                "label": label,
                "confidence": conf,
                "incidentType": incident_type,
                "bbox": (x1, y1, x2, y2),
                "camera_id": camera_id,
                "camera_config": camera_config,
                "detection_time": get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                "detected_by": SYSTEM_INFO["current_user"]
            })

        # Process stationary vehicles
        stationary_count = vehicle_tracker.update_tracks(
            camera_id, 
            current_vehicles, 
            perf_monitor.frame_counts.get(camera_id, 0)
        )

        # Add wrong-lane detection if lane_directions are available
        if "lane_directions" in camera_config:
            wrong_way_vehicles = vehicle_tracker.detect_wrong_way_vehicles(
                camera_id, 
                camera_config["lane_directions"]
            )
            
            for vehicle_id, bbox in wrong_way_vehicles:
                current_time = time.time()
                if "wrong_way_vehicle" not in last_sent[camera_id] or \
                   (current_time - last_sent[camera_id]["wrong_way_vehicle"]) >= COOLDOWN_PERIOD:
                    last_sent[camera_id]["wrong_way_vehicle"] = current_time
                    
                    alert_message = f"""
⚠️ VEHICLE DRIVING FROM WRONG SIDE DETECTED ⚠️
Camera: {camera_id}
Location: {camera_config['camera_location']}
Chainage: {camera_config['chainage']}
Time: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')} IST
Vehicle ID: {vehicle_id}
"""
                    print("\n" + "="*50)
                    print(alert_message)
                    print("="*50 + "\n")
                    
                    frame_detections.append({
                        "label": "wrong_way_vehicle",
                        "confidence": conf,
                        "incidentType": "Wrong_Way_Driving",
                        "bbox": bbox,
                        "camera_id": camera_id,
                        "camera_config": camera_config,
                        "detection_time": get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                        "detected_by": SYSTEM_INFO["current_user"],
                        "vehicle_id": vehicle_id,
                        "description": f"Vehicle detected driving from wrong side at {camera_config['camera_location']}"
                    })

        
        

        # Add congestion detection if needed
        if stationary_count >= CONGESTION_THRESHOLD:
            current_time = time.time()
            if "congestion" not in last_sent[camera_id] or \
               (current_time - last_sent[camera_id]["congestion"]) >= COOLDOWN_PERIOD:
                last_sent[camera_id]["congestion"] = current_time
                
                # Add prominent congestion alert
                alert_message = f"""
⚠️ TRAFFIC CONGESTION DETECTED ⚠️
Camera: {camera_id}
Location: {camera_config['camera_location']}
Chainage: {camera_config['chainage']}
Time: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')} IST
Stationary Vehicles: {stationary_count}
Threshold: {CONGESTION_THRESHOLD}
"""
                print("\n" + "="*50)
                print(alert_message)
                print("="*50 + "\n")
                
                frame_detections.append({
                    "label": "congestion",
                    "confidence": conf,
                    "incidentType": "Traffic_Congestion",
                    "bbox": (0, 0, frame.shape[1], frame.shape[0]),
                    "camera_id": camera_id,
                    "camera_config": camera_config,
                    "detection_time": get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                    "detected_by": SYSTEM_INFO["current_user"],
                    "stationary_count": stationary_count,
                    "description": f"Traffic congestion detected with {stationary_count} stationary vehicles"
                })

        # After updating tracks and obtaining stationary_count, add this section to process stationary vehicle alerts:
        if camera_id in vehicle_tracker.tracked_vehicles:
            for vehicle_id, (tracked_box, first_seen, is_stationary) in vehicle_tracker.tracked_vehicles[camera_id].items():
                # Only process vehicles that belong to VEHICLE_CLASSES
                # (If needed, you can add an extra check to verify the detection label)
                if is_stationary:
                    current_time = time.time()
                    if "stationary_vehicle" not in last_sent[camera_id] or \
                       (current_time - last_sent[camera_id]["stationary_vehicle"]) >= COOLDOWN_PERIOD:
                        last_sent[camera_id]["stationary_vehicle"] = current_time
                        alert_message = f"""
⚠️ STOPPED VEHICLE ALERT ⚠️
Camera: {camera_id}
Vehicle ID: {vehicle_id}
Location: {camera_config.get('camera_location', 'Unknown')}
Time: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')} IST
"""
                        print("\n" + "="*50)
                        print(alert_message)
                        print("="*50 + "\n")
                        
                        frame_detections.append({
                            "label": "is_stationary",
                            "confidence": conf,  # Confidence could be added if available
                            "incidentType": "Stopped_Vehicle",
                            "bbox": tracked_box,
                            "camera_id": camera_id,
                            "camera_config": camera_config,
                            "detection_time": get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                            "detected_by": SYSTEM_INFO["current_user"],
                            "vehicle_id": vehicle_id,
                            "description": f"Stationary vehicle detected in camera {camera_id}"
                        })

        # PROCESS CUSTOM DETECTIONS
        for box in results_custom[i].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            if cls not in CUSTOM_CLASSES:
                continue

            label = CUSTOM_CLASSES[cls]
            # Use the class-specific threshold if available
            threshold = CUSTOM_CONF_THRESHOLDS.get(label, CONF_THRESH)
            print(f"Custom Detection: label={label}, conf={conf:.2f}, threshold={threshold:.2f}")
            if conf < threshold:
                continue

            incident_type = get_incident_type(label)
            current_time = time.time()
            if label in last_sent[camera_id] and (current_time - last_sent[camera_id][label]) < COOLDOWN_PERIOD:
                continue
            last_sent[camera_id][label] = current_time

            # Special handling for "accident" detections
            if label == "accident":
                print(f"\n[DEBUG] Processing accident detection with confidence {conf:.2f}")
                stopped = False
                matching_vehicle = None
                if camera_id in vehicle_tracker.tracked_vehicles:
                    for vehicle_id, (tracked_box, _, _) in vehicle_tracker.tracked_vehicles[camera_id].items():
                        iou = VehicleTracker.calculate_iou((x1, y1, x2, y2), tracked_box)
                        print(f"[DEBUG] IOU with vehicle {vehicle_id}: {iou:.2f}")
                        if iou > IOU_THRESHOLD:
                            if vehicle_id in vehicle_tracker.vehicle_directions[camera_id]:
                                direction_vector, _ = vehicle_tracker.vehicle_directions[camera_id][vehicle_id]
                                velocity = math.hypot(direction_vector[0], direction_vector[1])
                                print(f"[DEBUG] Vehicle {vehicle_id} velocity: {velocity:.3f}")
                                if velocity < STATIONARY_VELOCITY_THRESHOLD:
                                    stopped = True
                                    matching_vehicle = vehicle_id
                                    print(f"[DEBUG] Vehicle {vehicle_id} is stopped.")
                                    break
                if not stopped:
                    print("[DEBUG] No stopped vehicle found for accident detection. Skipping.")
                    continue

                # Check accident confidence threshold
                if conf > 0.7:
                    accident_chance = 75  # Start with 75% chance
                    print("[DEBUG] Accident confidence > 0.7, initial chance set to 70%.")
                    # Look for concurrent fire or smoke detections with high confidence
                    fire_smoke_found = False
                    for other_box in results_custom[i].boxes:
                        other_label = CUSTOM_CLASSES.get(int(other_box.cls[0]))
                        other_conf = other_box.conf[0].item()
                        if other_label in ["fire", "smoke"]:
                            print(f"[DEBUG] Found {other_label} detection with confidence {other_conf:.2f}")
                            if other_conf > 0.7:
                                fire_smoke_found = True
                                break
                    if fire_smoke_found:
                        accident_chance = 85  # Upgrade to 85% chance
                        print("[DEBUG] Accident chance upgraded to 85% due to fire/smoke detection.")

                    alert_message = f"""
        ⚠️ ACCIDENT DETECTED ⚠️
        Camera: {camera_id}
        Time: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')} IST
        Accident Confidence: {conf:.2f}
        Calculated Accident Chance: {accident_chance}%
        Vehicle ID: {matching_vehicle}
                    """
                    print("\n" + "="*50)
                    print(alert_message)
                    print("="*50 + "\n")
                    
                    frame_detections.append({
                        "label": label,
                        "confidence": conf,
                        "incidentType": get_incident_type(label),
                        "bbox": (x1, y1, x2, y2),
                        "camera_id": camera_id,
                        "camera_config": camera_config,
                        "detection_time": get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                        "detected_by": SYSTEM_INFO["current_user"],
                        "accident_chance": accident_chance,
                        "description": f"Accident detected with a {accident_chance}% chance based on model confidence ({conf:.2f}) and vehicle stop detection."

                    })
                    continue  # Skip normal processing for accident detection

            # Process other custom detections normally
            incident_type = get_incident_type(label)
            frame_detections.append({
                "label": label,
                "confidence": conf,
                "incidentType": incident_type,
                "bbox": (x1, y1, x2, y2),
                "camera_id": camera_id,
                "camera_config": camera_config,
                "detection_time": get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                "detected_by": SYSTEM_INFO["current_user"]
            })
                      

        # Inside batch_process_frames, after processing each frame:
        for i in range(len(frames)):
            frame, timestamp = frames[i]
            camera_config = camera_configs[i % len(camera_configs)]
            camera_id = camera_config["camera_id"]
            current_time = time.time()

            # Check weather conditions
            is_bad_weather, condition_type, severity = detect_weather_condition(frame)
            if is_bad_weather:
                if ("weather_alert" not in last_sent[camera_id] or 
                    (current_time - last_sent[camera_id]["weather_alert"]) >= COOLDOWN_PERIOD):
                    
                    last_sent[camera_id]["weather_alert"] = current_time
                    
                    weather_message = f"""
⚠️ LOW VISIBILITY DUE TO {condition_type.upper()} DETECTED ⚠️
Camera: {camera_id}
Location: {camera_config.get('camera_location', 'Unknown')}
Chainage: {camera_config.get('chainage', 'Unknown')}
Time: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S')} IST
Severity: {severity}
"""
                    print("\n" + "="*50)
                    print(weather_message)
                    print("="*50 + "\n")

                    frame_detections.append({
                        "label": "Low_Visibility",
                        "confidence": conf,
                        "incidentType": "Weather_Condition_Alert",
                        "bbox": (0, 0, frame.shape[1], frame.shape[0]),
                        "camera_id": camera_id,
                        "camera_config": camera_config,
                        "detection_time": get_ist_time().strftime("%Y-%m-%d %H:%M:%S"),
                        "detected_by": SYSTEM_INFO["current_user"],
                        "weather_condition": condition_type,
                        "severity": severity,
                        "description": f"Low visibility detected due to {condition_type} conditions"
                    })


        all_detections.append(frame_detections)

    # Update performance metrics
    processing_time = time.time() - start_time
    for i, (_, timestamp) in enumerate(frames):
        camera_id = camera_configs[i % len(camera_configs)]["camera_id"]
        perf_monitor.update_stats(camera_id, processing_time / len(frames))

    return all_detections

incident_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def capture_frame_evidence(frame, camera_id, incident_type, suffix=""):
    """Save a frame as image evidence with an optional suffix for uniqueness"""
    try:
        evidence_dir = os.path.join("evidence", f"camera_{camera_id}")
        os.makedirs(evidence_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(evidence_dir, f"{incident_type}_{timestamp}{suffix}.jpg")
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
        success = cv2.imwrite(filename, frame, encode_params)
        if success:
            return filename
        return None
    except Exception as e:
        print(f"Error saving evidence for camera {camera_id}: {str(e)}")
        return None

# def process_detection_results(frames, all_detections, streams, incident_executor, handle_incident):
#     """
#     Process detection results using annotated image and video evidences.
    
#     This function:
#       1. Annotates a detection’s bounding box and label on the image.
#       2. Uses that image for immediate evidence.
#       3. Looks up the RTSPStream instance from 'streams' based on camera_id.
#       4. Extracts recent frames from the stream’s buffer.
#       5. Re-annotates each of those video frames with the same bounding box and label.
#       6. Creates a video evidence clip.
#       7. Submits an incident using two image evidences and one video evidence.
#     """
#     for i, detections in enumerate(all_detections):
#         if not detections:
#             continue
        
#         frame, timestamp = frames[i]
        
#         for detection in detections:
#             try:
#                 camera_config = detection["camera_config"]
#                 camera_id = str(camera_config["camera_id"])
#                 camera_location = str(camera_config.get("camera_location", "Unknown"))
#                 chainage = str(camera_config.get("chainage", "Unknown"))
#                 label = str(detection["label"])
#                 incident_type = str(detection["incidentType"])
#                 detection_time = str(detection["detection_time"])
#                 description = str(
#                     detection.get("description", f"Detected {label} ({incident_type}) at {camera_location}")
#                 )
                
#                 # Create an annotated frame for image evidence
#                 annotated_frame = draw_lane_regions(frame.copy(), camera_config)
#                 x1, y1, x2, y2 = detection["bbox"]
#                 color = (0, 255, 255)  # Neon yellow-green (cyan) color
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(annotated_frame, label, (x1, y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                
#                 # Capture two image evidences with different suffixes
#                 image_evidence_path1 = capture_frame_evidence(annotated_frame, camera_id, incident_type, suffix="_img1")
#                 image_evidence_path2 = capture_frame_evidence(annotated_frame, camera_id, incident_type, suffix="_img2")
#                 image_evidences = [image_evidence_path1, image_evidence_path2]
                
#                 # Lookup stream instance for this camera from the streams list
#                 current_stream = None
#                 for stream_obj, config in streams:
#                     if str(config["camera_id"]) == camera_id:
#                         current_stream = stream_obj
#                         break
                
#                 if current_stream is not None:
#                     # Get recent frames from the stream's buffer for video evidence
#                     video_frames = current_stream.get_recent_frames(duration=4, fps=15)
#                 else:
#                     print(f"Warning: no stream found for camera {camera_id}")
#                     video_frames = []
                
#                 # Re-annotate each video frame with bounding box & label
#                 annotated_video_frames = []
#                 for frame_vid in video_frames:
#                     annotated_vid = frame_vid.copy()
#                     cv2.rectangle(annotated_vid, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(annotated_vid, label, (x1, y1 - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#                     annotated_video_frames.append(annotated_vid)
                
#                 video_evidence_path = create_video_evidence(annotated_video_frames, camera_id, incident_type)
                
#                 if all(image_evidences) and video_evidence_path:
#                     print(f"Processing detection for Camera {camera_id}: {incident_type} at {camera_location}")
#                     incident_executor.submit(
#                         handle_incident,
#                         "____",                      # vehicle_number placeholder
#                         camera_location,
#                         chainage,
#                         label,
#                         incident_type,
#                         description,
#                         camera_id,
#                         str(camera_config.get("rtsp_url", "")),
#                         detection_time,
#                         image_evidences,             # Pass both image evidences
#                         video_evidence_path          # Video evidence file path
#                     )
#                 else:
#                     print(f"⚠️ Camera {camera_id}: Failed to save some evidence. Skipping incident report.")
                    
#             except Exception as e:
#                 print(f"Error processing detection in main_onnx: {str(e)}")
#                 continue

def process_detection_results(frames, all_detections, streams, incident_executor, handle_incident):
    """
    Process detection results using annotated image and video evidences.
    
    This function:
      1. Annotates a detection’s bounding box and label on the image.
      2. Uses that image for immediate evidence.
      3. Looks up the RTSPStream instance from 'streams' based on camera_id.
      4. Extracts recent frames from the stream’s buffer.
      5. Re-annotates each of those video frames with the same bounding box and label.
      6. Creates a video evidence clip.
      7. Submits an incident using two image evidences and one video evidence.
    """
    for i, detections in enumerate(all_detections):
        if not detections:
            continue
        
        frame, timestamp = frames[i]
        
        for detection in detections:
            try:
                camera_config = detection["camera_config"]
                camera_id = str(camera_config["camera_id"])
                camera_location = str(camera_config.get("camera_location", "Unknown"))
                chainage = str(camera_config.get("chainage", "Unknown"))
                label = str(detection["label"])
                incident_type = str(detection["incidentType"])
                detection_time = str(detection["detection_time"])
                # Get the confidence value from the detection dictionary
                confidence = detection.get("confidence", 0.0)
                description = str(
                    detection.get("description", f"Detected {label} ({incident_type}) at {camera_location}. Confidence: {confidence:.2f}")
                )
                
                # Create an annotated frame for image evidence
                annotated_frame = draw_lane_regions(frame.copy(), camera_config)
                x1, y1, x2, y2 = detection["bbox"]
                color = (0, 255, 255)  # Neon yellow-green (cyan) color
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Capture two image evidences with different suffixes
                image_evidence_path1 = capture_frame_evidence(annotated_frame, camera_id, incident_type, suffix="_img1")
                image_evidence_path2 = capture_frame_evidence(annotated_frame, camera_id, incident_type, suffix="_img2")
                image_evidences = [image_evidence_path1, image_evidence_path2]
                
                # Lookup stream instance for this camera from the streams list
                current_stream = None
                for stream_obj, config in streams:
                    if str(config["camera_id"]) == camera_id:
                        current_stream = stream_obj
                        break
                
                if current_stream is not None:
                    # Get recent frames from the stream's buffer for video evidence
                    video_frames = current_stream.get_recent_frames(duration=4, fps=15)
                else:
                    print(f"Warning: no stream found for camera {camera_id}")
                    video_frames = []
                
                # Re-annotate each video frame with bounding box & label
                annotated_video_frames = []
                for frame_vid in video_frames:
                    annotated_vid = frame_vid.copy()
                    cv2.rectangle(annotated_vid, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_vid, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    annotated_video_frames.append(annotated_vid)
                
                video_evidence_path = create_video_evidence(annotated_video_frames, camera_id, incident_type)
                
                # Pass the confidence value as an extra argument to the handler.
                if all(image_evidences) and video_evidence_path:
                    print(f"Processing detection for Camera {camera_id}: {incident_type} at {camera_location}. Confidence: {confidence:.2f}")
                    incident_executor.submit(
                        handle_incident,
                        "____",                      # vehicle_number placeholder
                        camera_location,
                        chainage,
                        label,
                        incident_type,
                        description,
                        confidence,                  # Extra parameter for detection confidence
                        camera_id,
                        str(camera_config.get("rtsp_url", "")),
                        detection_time,
                        image_evidences,             # Pass both image evidences
                        video_evidence_path          # Video evidence file path
                    )
                else:
                    print(f"⚠️ Camera {camera_id}: Failed to save some evidence. Skipping incident report.")
                    
            except Exception as e:
                print(f"Error processing detection in main_onnx: {str(e)}")
                continue

def cleanup_resources():
    """Cleanup system resources"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    incident_executor.shutdown()  # Shut down the thread pool
    print(f"\nResource cleanup completed at {format_ist_time()} IST")
    print(f"User session: {SYSTEM_INFO['current_user']}")

def main():
    """Main function with improved error handling"""
    try:
        print(f"Starting system at {format_ist_time()} IST")
        print(f"User: {SYSTEM_INFO['current_user']}")
        
        # Initialize components
        models = load_models()
        vehicle_tracker = VehicleTracker()
        perf_monitor = PerformanceMonitor()
        
        # Initialize streams
        camera_sources = CAMERA_CONFIGS if SOURCE_TYPE == "rtsp" else MP4_SOURCES
        streams = []
        failed_streams = []
        
        for camera_config in camera_sources:
            try:
                resolution = camera_config.get("resolution", CAMERA_RESOLUTION)
                url = camera_config["rtsp_url"] if SOURCE_TYPE == "rtsp" else camera_config["mp4_path"]
                camera_id = camera_config["camera_id"]
                
                print(f"\nInitializing camera {camera_id}...")
                stream = RTSPStream(
                    url=url,
                    camera_id=camera_id,
                    frame_skip=FRAME_SKIP,
                    resolution=resolution
                )
                # Wait briefly to capture a sample frame for lane detection
                time.sleep(2)
                grabbed, sample_frame, _ = stream.read()
                if grabbed:
                    # Use auto-detection to set lane regions
                    camera_config["lane_directions"] = auto_detect_lane_regions(sample_frame, num_lanes=2)
                    print(f"Auto-detected lanes for Camera {camera_id}: {camera_config['lane_directions']}")
                else:
                    # Fallback if no frame captured: use default definition
                    camera_config["lane_directions"] = define_lane_directions(resolution[0], resolution[1], num_lanes=2, horizontal=True)
                    print(f"Used default lane definition for Camera {camera_id}")
                
                streams.append((stream, camera_config))
                time.sleep(1)
                
            except Exception as e:
                print(f"Failed to initialize camera {camera_config['camera_id']}: {str(e)}")
                failed_streams.append(camera_config['camera_id'])
        
        if not streams:
            raise RuntimeError("No cameras could be initialized")
        
        if failed_streams:
            print(f"\nWarning: Failed to initialize cameras: {', '.join(map(str, failed_streams))}")
        
        print(f"\nSuccessfully initialized {len(streams)} cameras")
        print(f"Processing started at {format_ist_time()} IST")
        
        running = True
        while running:
            frames = []
            frame_configs = []
            active_streams = 0
            
            # Collect frames from all streams
            for stream, config in streams:
                grabbed, frame, timestamp = stream.read()
                if grabbed:
                    frames.append((frame, timestamp))
                    frame_configs.append(config)
                    active_streams += 1
            
            # Process frames in batches
            if frames:
                for i in range(0, len(frames), BATCH_SIZE):
                    batch_frames = frames[i:i+BATCH_SIZE]
                    batch_configs = frame_configs[i:i+BATCH_SIZE]
                    
                    try:
                        detections = batch_process_frames(models, batch_frames, batch_configs, vehicle_tracker, perf_monitor)
                        process_detection_results(batch_frames, detections, streams, incident_executor, handle_incident)
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")
                        continue
            # # Display frames from all streams
            # for config, (frame, _) in zip(frame_configs, frames):
            #     annotated_frame = draw_lane_regions(frame.copy(), config)
            #     cv2.imshow(f"Camera {config['camera_id']}", annotated_frame)
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        print(f"\nCritical error occurred: {str(e)}")
    finally:
        # Cleanup
        print("\nCleaning up resources...")
        for stream, _ in streams:
            stream.stop()
        cleanup_resources()
        print(f"\nSystem shutdown completed at {format_ist_time()} IST")

if __name__ == "__main__":
    main()
