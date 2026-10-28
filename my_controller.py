import base64
import csv
import json
import math
import re
import sys
import threading
import time
import urllib.request
import cv2
import numpy as np
from controller import Robot
from ultralytics import YOLO

# --- 1. TURTLEBOT3 KINEMATIC & FUSION CONSTANTS ---
WHEEL_RADIUS = 0.033       # meters
AXLE_LENGTH = 0.160        # meters (TurtleBot3Burger wheel track)
MAX_WHEEL_VEL = 6.67       # rad/s (Dynamixel hardware limit)

CRUISE_LINEAR_VEL = 0.14   # m/s forward exploration speed
APPROACH_LINEAR_VEL = 0.11 # m/s target approach speed
EVADE_LINEAR_VEL = -0.04   # m/s slight reverse during initial corner unwedging
EVADE_ANGULAR_VEL = 1.35   # rad/s for obstacle evasion spins
SWEEP_ANGULAR_VEL = 0.80   # rad/s for room scanning

# Distance & Safety Thresholds (Tightened Turn Distances)
EMERGENCY_DIST = 0.12      # Dead-ahead turn trigger (12 cm)
SHOULDER_DIST = 0.17       # Angled/corner turn trigger (17 cm)
GOAL_STOP_DIST = 0.20      # Metric target arrival stop (20 cm)

# Camera & Visual Servoing Parameters
FRAME_CENTER_X = 320.0
CAMERA_HFOV = 1.047        # ~60 degrees horizontal field of view in radians
CENTER_TOLERANCE = 70.0    # Pixel window for centered target stop
KP_YAW = 0.0038            # Proportional steering gain (rad/s per pixel error)

# Strict Target Classes Only
TARGET_CLASSES = {"fire hydrant", "sports ball", "stop sign"}
# Close-range cylindrical classes that only pass if HSV confirms a solid crimson surface
AMBIGUOUS_CYLINDER_CLASSES = {"traffic light", "parking meter", "fire hydrant"}

# --- 2. 2D OCCUPANCY GRID MAP CONFIGURATION ---
MAP_SIZE = 480             # 480x480 pixel mental map (matches 480px camera height)
MAP_SCALE = 110.0          # 110 pixels per meter (~4.3m x 4.3m arena coverage)
MAP_CENTER = MAP_SIZE // 2

occ_grid = np.full((MAP_SIZE, MAP_SIZE), 0.5, dtype=np.float32)
trajectory_pts = []

# --- 3. DUAL-MODE OLLAMA VLM (EXPLORATION + TARGET VERIFICATION) ---
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
VLM_MODEL = "moondream"
VLM_POLL_INTERVAL = 2.0

VLM_EXPLORE_PROMPT = (
    "Describe the robot path ahead as DEAD_END, DOORWAY, or OPEN_ROOM."
)

VLM_VERIFY_PROMPT = (
    "Is there a red fire hydrant in this image? Answer yes or no."
)

STATE_PATTERNS = {
    "DEAD_END": re.compile(r"\b(DEAD[\s_-]*END|WALL|BLOCKED|CORNER|OBSTACLE)\b", re.IGNORECASE),
    "DOORWAY":  re.compile(r"\b(DOOR(?:WAY)?|GAP|NARROW|BETWEEN|PASSAGE|SLIT)\b", re.IGNORECASE),
    "OPEN_ROOM": re.compile(r"\b(OPEN[\s_-]*ROOM|OPEN|FLOOR|CLEAR|EMPTY)\b", re.IGNORECASE),
}


def compute_crimson_ratio(bgr_roi: np.ndarray) -> float:
    """Computes the fraction of pixels matching the fire hydrant's crimson/red HSV profile."""
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 95, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([168, 95, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    return float(np.count_nonzero(red_mask)) / float(red_mask.size)


def parse_vlm_explore(raw_text: str, front_m: float, shoulder_m: float) -> str:
    if raw_text and isinstance(raw_text, str):
        for state_name, pattern in STATE_PATTERNS.items():
            if pattern.search(raw_text):
                return state_name
    # Scene-grounded fallback when Moondream outputs a generic caption on OpenGL textures
    if front_m < 0.35:
        return "DEAD_END"
    if shoulder_m < 0.26 and front_m > 0.45:
        return "DOORWAY"
    return "OPEN_ROOM"


def parse_vlm_verify(raw_text: str, frame_bgr: np.ndarray, has_visual_lock: bool, dist_to_pin: float) -> str:
    if raw_text and isinstance(raw_text, str):
        text = raw_text.strip().upper()
        has_yes = bool(re.search(r"\b(YES|YEAH|TRUE|DEFINITELY|CORRECT|VISIBLE|PRESENT)\b", text))
        has_no = bool(re.search(r"\b(NO|NOT|CANNOT|NONE|FALSE|EMPTY)\b", text))

        if has_yes and not has_no:
            return "VERIFIED_YES"
        if any(k in text for k in ["HYDRANT", "RED", "FIRE", "CYLINDER"]) and not has_no:
            return "VERIFIED_YES"
        if has_no and not has_yes and not has_visual_lock:
            return "REJECTED_NO"

    # Grounded visual verification if Ollama returns empty string after vision encode
    crimson_score = compute_crimson_ratio(frame_bgr)
    if has_visual_lock or crimson_score >= 0.015:
        return "VERIFIED_YES"
    if dist_to_pin < 0.48 and crimson_score < 0.005:
        return "REJECTED_NO"
    return "UNVERIFIED"


vlm_lock = threading.Lock()
latest_bgr_frame = None
vlm_verify_mode = False
vlm_target_hint = False
vlm_dist_to_pin_hint = 3.5
vlm_front_dist_hint = 3.5
vlm_shoulder_dist_hint = 3.5
vlm_environment_state = "OPEN_ROOM"
vlm_verify_state = "UNVERIFIED"
vlm_latency_ms = 0.0
vlm_query_count = 0
controller_running = True


def vlm_background_worker():
    global vlm_environment_state, vlm_verify_state, vlm_latency_ms, vlm_query_count
    time.sleep(0.5)
    while controller_running:
        with vlm_lock:
            frame_copy = latest_bgr_frame.copy() if latest_bgr_frame is not None else None
            verify_active = vlm_verify_mode
            target_hint = vlm_target_hint
            pin_dist_hint = vlm_dist_to_pin_hint
            front_hint = vlm_front_dist_hint
            shoulder_hint = vlm_shoulder_dist_hint
        if frame_copy is None:
            time.sleep(0.2)
            continue

        small_frame = cv2.resize(frame_copy, (320, 240))
        success, buffer = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            b64_image = base64.b64encode(buffer).decode("utf-8")
            prompt_to_use = VLM_VERIFY_PROMPT if verify_active else VLM_EXPLORE_PROMPT
            # Omit num_predict so Ollama does not truncate after the 729 image tokens
            payload = {
                "model": VLM_MODEL,
                "prompt": prompt_to_use,
                "images": [b64_image],
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0.0}
            }
            t_start = time.perf_counter()
            try:
                req = urllib.request.Request(
                    OLLAMA_URL,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=20.0) as resp:
                    result_json = json.loads(resp.read().decode("utf-8"))
                    raw_reply = result_json.get("response", "").strip()
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                with vlm_lock:
                    if verify_active:
                        parsed_v = parse_vlm_verify(raw_reply, small_frame, target_hint, pin_dist_hint)
                        if vlm_verify_state != "VERIFIED_YES" and parsed_v == "VERIFIED_YES":
                            vlm_verify_state = "VERIFIED_YES"
                            print(f"[VLM VERIFY] Target Confirmed -> VERIFIED_YES ({elapsed_ms:.0f}ms)")
                        elif vlm_verify_state != "VERIFIED_YES":
                            vlm_verify_state = parsed_v
                    else:
                        new_state = parse_vlm_explore(raw_reply, front_hint, shoulder_hint)
                        if new_state != vlm_environment_state:
                            print(f"[VLM EXPLORE] Scene State -> {new_state} ({elapsed_ms:.0f}ms)")
                        vlm_environment_state = new_state
                    vlm_latency_ms = elapsed_ms
                    vlm_query_count += 1
            except Exception:
                with vlm_lock:
                    if not verify_active:
                        vlm_environment_state = "OPEN_ROOM"
        time.sleep(VLM_POLL_INTERVAL)


def wrap_angle(angle: float) -> float:
    if math.isnan(angle) or math.isinf(angle):
        return 0.0
    return math.atan2(math.sin(angle), math.cos(angle))


def world_to_map(wx: float, wy: float):
    if math.isnan(wx) or math.isnan(wy) or math.isinf(wx) or math.isinf(wy):
        return MAP_CENTER, MAP_CENTER
    u = int(np.clip(MAP_CENTER + wx * MAP_SCALE, 0, MAP_SIZE - 1))
    v = int(np.clip(MAP_CENTER - wy * MAP_SCALE, 0, MAP_SIZE - 1))
    return u, v


def compute_wheel_speeds(linear_v: float, angular_w: float):
    vel_left = (linear_v - (angular_w * AXLE_LENGTH / 2.0)) / WHEEL_RADIUS
    vel_right = (linear_v + (angular_w * AXLE_LENGTH / 2.0)) / WHEEL_RADIUS
    vel_left = float(np.clip(vel_left, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
    vel_right = float(np.clip(vel_right, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
    return vel_left, vel_right


def detect_red_hydrant_signature(bgr_img: np.ndarray, min_ratio: float = 0.22):
    """
    Verifies whether the close-range object ahead has the fire hydrant's solid
    crimson surface and returns (is_red_hydrant, pixel_error, bounding_rect, box_h).
    """
    h, w, _ = bgr_img.shape
    x0, y0, x1, y1 = int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.92)
    roi = bgr_img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 95, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([168, 95, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)
    if red_ratio >= min_ratio:
        moments = cv2.moments(red_mask)
        if moments["m00"] > 0:
            cx_roi = moments["m10"] / moments["m00"]
            cx_full = cx_roi + x0
            ys, xs = np.where(red_mask > 0)
            bx1, by1 = int(np.min(xs) + x0), int(np.min(ys) + y0)
            bx2, by2 = int(np.max(xs) + x0), int(np.max(ys) + y0)
            return True, float(cx_full - FRAME_CENTER_X), (bx1, by1, bx2, by2), float(by2 - by1)
    return False, 0.0, None, 0.0


# --- 4. INITIALIZE ROBOT, ENCODERS, LIDAR & CAMERA ---
model = YOLO("yolov8n.pt")

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_sensor = robot.getDevice('left wheel sensor')
right_sensor = robot.getDevice('right wheel sensor')
left_sensor.enable(timestep)
right_sensor.enable(timestep)

vlm_thread = threading.Thread(target=vlm_background_worker, daemon=True)
vlm_thread.start()

# Robot Pose in World Frame (Starts at origin facing +Y [UP on map])
robot_x = 0.0
robot_y = 0.0
robot_theta = math.pi / 2.0
prev_left_enc = None
prev_right_enc = None

# Persistent Target Memory, Phantom-Pin Counter & Object Permanence
goal_pinned = False
goal_x = 0.0
goal_y = 0.0
phantom_pin_timer = 0
pins_cleared_count = 0
target_lock_timer = 0
last_pixel_error = 0.0
last_box_h = 0.0

# 3-Stage Obstacle Bypass State Machine
turn_timer = 0
bypass_cruise_timer = 0
bypass_reorient_timer = 0
turn_direction = 1.0
search_timer = 0
goal_latched = False
report_printed = False

mission_start_time = time.perf_counter()
total_steps = 0
tier_counts = {"P1_SURVIVAL": 0, "P2_MISSION": 0, "P3_EXPLORATION": 0}
yolo_latencies_ms = []
vlm_latencies_ms = []
telemetry_rows = []

# --- 5. MAIN CONTROL & MAPPING LOOP ---
while robot.step(timestep) != -1:
    loop_start = time.perf_counter()
    total_steps += 1

    # A. Wheel Odometry Pose Update (x, y, theta)
    cur_l = left_sensor.getValue()
    cur_r = right_sensor.getValue()
    if not (math.isnan(cur_l) or math.isnan(cur_r)):
        if prev_left_enc is None:
            prev_left_enc, prev_right_enc = cur_l, cur_r
        dl = (cur_l - prev_left_enc) * WHEEL_RADIUS
        dr = (cur_r - prev_right_enc) * WHEEL_RADIUS
        prev_left_enc, prev_right_enc = cur_l, cur_r

        dc = (dl + dr) / 2.0
        dtheta = (dr - dl) / AXLE_LENGTH
        robot_x += dc * math.cos(robot_theta + dtheta / 2.0)
        robot_y += dc * math.sin(robot_theta + dtheta / 2.0)
        robot_theta = wrap_angle(robot_theta + dtheta)

    rob_u, rob_v = world_to_map(robot_x, robot_y)
    if len(trajectory_pts) == 0 or math.hypot(rob_u - trajectory_pts[-1][0], rob_v - trajectory_pts[-1][1]) > 3:
        trajectory_pts.append((rob_u, rob_v))

    # B. Read 360 LiDAR & Update 2D Occupancy Grid Map in Memory
    raw_lidar = lidar.getRangeImage()
    num_rays = 360
    if raw_lidar:
        ranges = np.array(raw_lidar, dtype=np.float32)
        num_rays = len(ranges)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=3.5)
        ranges[ranges < 0.10] = 3.5

        mid = num_rays // 2
        boresight_arc = ranges[mid - 6 : mid + 6]
        left_shoulder_arc = ranges[mid - 52 : mid - 22]
        right_shoulder_arc = ranges[mid + 22 : mid + 52]
        left_sector = ranges[mid - 85 : mid - 20]
        right_sector = ranges[mid + 20 : mid + 85]

        target_metric_dist = float(np.min(boresight_arc))
        closest_shoulder_obstacle = float(min(np.min(left_shoulder_arc), np.min(right_shoulder_arc)))
        closest_front_obstacle = float(min(target_metric_dist, closest_shoulder_obstacle))
        left_clearance = float(np.mean(left_sector))
        right_clearance = float(np.mean(right_sector))

        # Raycast every 2nd LiDAR beam onto the 2D Occupancy Grid
        free_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        hit_pts = []
        for i in range(0, num_rays, 2):
            r = float(ranges[i])
            ray_angle = robot_theta + ((mid - i) / float(num_rays)) * (2.0 * math.pi)
            end_r = min(r, 3.0)
            wx = robot_x + end_r * math.cos(ray_angle)
            wy = robot_y + end_r * math.sin(ray_angle)
            eu, ev = world_to_map(wx, wy)
            cv2.line(free_mask, (rob_u, rob_v), (eu, ev), 1, 1)
            if r < 3.2:
                hit_pts.append((eu, ev))

        occ_grid[free_mask == 1] = np.clip(occ_grid[free_mask == 1] - 0.04, 0.05, 0.98)
        for hu, hv in hit_pts:
            occ_grid[max(0, hv-1):min(MAP_SIZE, hv+2), max(0, hu-1):min(MAP_SIZE, hu+2)] = np.clip(
                occ_grid[max(0, hv-1):min(MAP_SIZE, hv+2), max(0, hu-1):min(MAP_SIZE, hu+2)] + 0.18, 0.05, 0.98
            )
    else:
        ranges = np.full(360, 3.5, dtype=np.float32)
        mid = 180
        target_metric_dist = 3.5
        closest_shoulder_obstacle = 3.5
        closest_front_obstacle = 3.5
        left_clearance = 3.5
        right_clearance = 3.5

    # Distance to current Pinned Goal (used to gate close-range HSV checks)
    dist_to_pinned_goal = math.hypot(goal_x - robot_x, goal_y - robot_y) if goal_pinned else 3.5

    # C. Process Camera Frame (YOLOv8 + Close-Range HSV Crimson Verification)
    target_visible = False
    hsv_verified_lock = False
    pixel_error = 0.0
    annotated_frame = None
    img_bgr = None
    yolo_ms = 0.0

    with vlm_lock:
        current_vlm_state = vlm_environment_state
        current_vlm_verify = vlm_verify_state
        current_vlm_ms = vlm_latency_ms

    image_data = camera.getImage()
    if image_data:
        img_bgra = np.frombuffer(image_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

        t_yolo = time.perf_counter()
        results = model.predict(source=img_bgr, conf=0.15, verbose=False)
        yolo_ms = (time.perf_counter() - t_yolo) * 1000.0
        yolo_latencies_ms.append(yolo_ms)
        if current_vlm_ms > 0:
            vlm_latencies_ms.append(current_vlm_ms)

        annotated_frame = img_bgr.copy()
        h_img, w_img, _ = img_bgr.shape

        best_box = None
        best_score = 0.0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bx1, by1, bx2, by2 = max(0, int(x1)), max(0, int(y1)), min(w_img, int(x2)), min(h_img, int(y2))
            box_area = float((bx2 - bx1) * (by2 - by1))

            if cls_name in TARGET_CLASSES and conf_val >= 0.25:
                score = conf_val * 2.0 + (box_area / (w_img * h_img))
                if score > best_score:
                    best_score = score
                    best_box = (x1, y1, x2, y2, "fire hydrant", conf_val, False)

            # Only allow HSV cylinder rescue when shoulders are clear or robot is physically near the Pinned Goal
            elif cls_name in AMBIGUOUS_CYLINDER_CLASSES and target_metric_dist < 0.65 and (
                dist_to_pinned_goal < 0.55 or closest_shoulder_obstacle > 0.22
            ):
                crop = img_bgr[by1:by2, bx1:bx2]
                crimson_ratio = compute_crimson_ratio(crop)
                if crimson_ratio >= 0.35 and (by2 - by1) >= 260:
                    score = crimson_ratio + (box_area / (w_img * h_img))
                    if score > best_score:
                        best_score = score
                        best_box = (x1, y1, x2, y2, "fire hydrant [HSV]", crimson_ratio, True)

        if best_box is not None:
            x1, y1, x2, y2, label_str, score_val, is_hsv_box = best_box
            box_center_x = float((x1 + x2) / 2.0)
            box_h = float(max(y2 - y1, 10.0))
            last_box_h = box_h
            pixel_error = box_center_x - FRAME_CENTER_X
            target_visible = True
            hsv_verified_lock = is_hsv_box

            box_color = (0, 255, 255) if is_hsv_box else (255, 140, 0)
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(annotated_frame, f"{label_str} {score_val:.2f}", (int(x1), max(22, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2)

            if score_val >= 0.35 or is_hsv_box:
                cam_bearing = -(pixel_error / FRAME_CENTER_X) * (CAMERA_HFOV / 2.0)
                ray_idx = int(np.clip(mid - (cam_bearing / (2.0 * math.pi)) * num_rays, 2, num_rays - 3))
                center_ray_dist = float(np.min(ranges[ray_idx - 1 : ray_idx + 2]))
                bbox_est_dist = float(np.clip(185.0 / box_h, 0.18, 1.40))

                if center_ray_dist < 1.35 and closest_shoulder_obstacle > (SHOULDER_DIST + 0.04):
                    est_dist = center_ray_dist
                else:
                    est_dist = bbox_est_dist

                obs_gx = robot_x + est_dist * math.cos(robot_theta + cam_bearing)
                obs_gy = robot_y + est_dist * math.sin(robot_theta + cam_bearing)
                if not goal_pinned:
                    goal_x, goal_y = obs_gx, obs_gy
                    goal_pinned = True
                else:
                    goal_x = 0.70 * goal_x + 0.30 * obs_gx
                    goal_y = 0.70 * goal_y + 0.30 * obs_gy

    # Recompute distance and bearing to the Pinned Target in Memory
    if goal_pinned:
        dist_to_pinned_goal = math.hypot(goal_x - robot_x, goal_y - robot_y)
        world_angle_to_goal = math.atan2(goal_y - robot_y, goal_x - robot_x)
        rel_angle_to_goal = wrap_angle(world_angle_to_goal - robot_theta)

        boresight_wx = robot_x + target_metric_dist * math.cos(robot_theta)
        boresight_wy = robot_y + target_metric_dist * math.sin(robot_theta)
        boresight_dist_to_pin = math.hypot(boresight_wx - goal_x, boresight_wy - goal_y)
    else:
        dist_to_pinned_goal = 3.5
        rel_angle_to_goal = 0.0
        boresight_dist_to_pin = 3.5

    # Full-Frame Close-Range HSV Fallback (ONLY when near Pinned Goal or when shoulders are open)
    if not target_visible and annotated_frame is not None and target_metric_dist < 0.45 and (
        dist_to_pinned_goal < 0.50 or closest_shoulder_obstacle > 0.22
    ):
        is_red, red_err, red_rect, red_h = detect_red_hydrant_signature(img_bgr, min_ratio=0.22)
        if is_red and red_h >= 260.0:
            target_visible = True
            hsv_verified_lock = True
            pixel_error = red_err
            last_box_h = red_h
            rx1, ry1, rx2, ry2 = red_rect
            cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(annotated_frame, "CLOSE-RANGE HSV LOCK: fire hydrant", (rx1, max(22, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

    # Update VLM frame buffer & context hints AFTER YOLO/HSV
    if img_bgr is not None:
        with vlm_lock:
            if not goal_pinned or target_visible or dist_to_pinned_goal < 0.48:
                latest_bgr_frame = img_bgr
                vlm_target_hint = target_visible
            vlm_verify_mode = goal_pinned
            vlm_dist_to_pin_hint = dist_to_pinned_goal
            vlm_front_dist_hint = closest_front_obstacle
            vlm_shoulder_dist_hint = closest_shoulder_obstacle

    # --- PHANTOM-PIN INVALIDATION (CLEARS MOVED OR FALSE PINS) ---
    # Stage 1 and Stage 2 of box bypass are NEVER interrupted by keyhole sightings
    in_bypass = (turn_timer > 0) or (bypass_cruise_timer > 0) or (bypass_reorient_timer > 0)

    if goal_pinned and not target_visible and not in_bypass:
        looking_at_empty_pin = (dist_to_pinned_goal < 0.45 and abs(rel_angle_to_goal) < 0.35 and target_metric_dist > 0.35)
        vlm_rejected_pin = (current_vlm_verify == "REJECTED_NO" and abs(rel_angle_to_goal) < 0.30)

        if looking_at_empty_pin or vlm_rejected_pin:
            phantom_pin_timer += 1
        else:
            phantom_pin_timer = max(0, phantom_pin_timer - 1)

        if phantom_pin_timer >= 20:
            print(f"[MAP MEMORY] Cleared phantom/relocated pin at ({goal_x:.2f}m, {goal_y:.2f}m) -> Returning to P3 Exploration.")
            goal_pinned = False
            target_lock_timer = 0
            phantom_pin_timer = 0
            pins_cleared_count += 1
            with vlm_lock:
                vlm_verify_state = "UNVERIFIED"
                vlm_verify_mode = False
                vlm_target_hint = False
    else:
        phantom_pin_timer = 0

    # Cancel Stage 3 reorientation ONLY if target is visible AND shoulders are clear of the box gap
    if target_visible and closest_shoulder_obstacle > (SHOULDER_DIST + 0.03) and turn_timer == 0 and bypass_cruise_timer == 0:
        bypass_reorient_timer = 0
        in_bypass = False

    # Update Short-Term Object Permanence (Strictly suppressed during active box bypass)
    if in_bypass:
        target_lock_timer = 0
    elif target_visible:
        target_lock_timer = 45
        last_pixel_error = pixel_error
    elif target_lock_timer > 0:
        target_lock_timer -= 1
        pixel_error = last_pixel_error

    target_locked = (target_visible or (target_lock_timer > 0)) and not in_bypass

    if goal_latched and target_metric_dist > (GOAL_STOP_DIST + 0.08):
        goal_latched = False
        report_printed = False

    # D. Subsumption Arbitration with True Close-Range Scale Verification
    status_label = ""
    active_tier = ""
    chosen_steer_angle = 0.0

    # Verify the hydrant is genuinely right in front of the robot (NOT viewed from 0.75m through a narrow box slit!)
    target_is_truly_close = (
        (goal_pinned and dist_to_pinned_goal < 0.42 and boresight_dist_to_pin < 0.25)
        or (last_box_h >= 285.0)
        or hsv_verified_lock
    )

    obstacle_is_target = (
        target_locked
        and target_is_truly_close
        and abs(pixel_error) < 95.0
        and target_metric_dist < 0.35
    )

    shoulder_blocked = (closest_shoulder_obstacle < SHOULDER_DIST) and not obstacle_is_target
    boresight_blocked = (target_metric_dist < EMERGENCY_DIST) and not (
        target_locked and target_is_truly_close and abs(pixel_error) < CENTER_TOLERANCE
    )

    # STRICT GOAL STOP: Requires target_locked + truly close to target/pin (not at a box gap!) + centered + <= 0.20m
    arrived_at_goal = (
        target_locked
        and target_is_truly_close
        and abs(pixel_error) < CENTER_TOLERANCE
        and target_metric_dist <= GOAL_STOP_DIST
    )

    # Priority 2A: Parked at Verified Target Goal
    if goal_latched or arrived_at_goal:
        goal_latched = True
        turn_timer = 0
        bypass_cruise_timer = 0
        bypass_reorient_timer = 0
        left_speed, right_speed = 0.0, 0.0
        active_tier = "P2_MISSION"
        status_label = f"P2: GOAL REACHED! STOPPED AT {target_metric_dist:.2f}m"

        if not report_printed:
            report_printed = True
            total_time_s = time.perf_counter() - mission_start_time
            avg_yolo = float(np.mean(yolo_latencies_ms)) if yolo_latencies_ms else 0.0
            avg_vlm = float(np.mean(vlm_latencies_ms)) if vlm_latencies_ms else 0.0
            avg_fps = total_steps / total_time_s if total_time_s > 0 else 0.0

            print("\n" + "=" * 60)
            print("   AUTONOMOUS MAP-GUIDED NAVIGATION (v2.1) - MISSION REPORT")
            print("=" * 60)
            print(f" Total Mission Time       : {total_time_s:.2f} seconds ({total_steps} steps)")
            print(f" Final Target Stop Dist   : {target_metric_dist:.3f} meters (Goal <= {GOAL_STOP_DIST:.2f}m)")
            print(f" Pinned Goal World Coords : ({goal_x:.2f}m, {goal_y:.2f}m)")
            print(f" Phantom Pins Cleared     : {pins_cleared_count} pins invalidated")
            print(f" VLM Verification Status  : {current_vlm_verify} ({vlm_query_count} async queries)")
            print(f" Fast-Loop Control Rate   : {avg_fps:.1f} Hz")
            print(f" Avg YOLOv8n Latency      : {avg_yolo:.1f} ms")
            print(f" Avg Ollama VLM Latency   : {avg_vlm:.1f} ms")
            print("-" * 60)
            print(f" P1 (LiDAR Survival)      : {tier_counts['P1_SURVIVAL']} steps ({100*tier_counts['P1_SURVIVAL']/total_steps:.1f}%)")
            print(f" P2 (Map+YOLO+HSV Fusion) : {tier_counts['P2_MISSION']} steps ({100*tier_counts['P2_MISSION']/total_steps:.1f}%)")
            print(f" P3 (VLM Exploration)     : {tier_counts['P3_EXPLORATION']} steps ({100*tier_counts['P3_EXPLORATION']/total_steps:.1f}%)")
            print("=" * 60 + "\n")

            header = ["step", "tier", "robot_x", "robot_y", "front_lidar_m", "boresight_m", "vlm_state", "yolo_ms"]
            for fname in ["mission_telemetry.csv", f"telemetry_v2_{time.strftime('%H%M%S')}.csv"]:
                with open(fname, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(telemetry_rows)
            print("[INFO] Saved telemetry to mission_telemetry.csv (and timestamped backup)")

    # Priority 1A (Stage 1 of Bypass): Evasive Pivot Away from Narrow Box Gap
    elif turn_timer > 0:
        left_speed, right_speed = compute_wheel_speeds(EVADE_LINEAR_VEL, turn_direction * EVADE_ANGULAR_VEL)
        turn_timer -= 1
        active_tier = "P1_SURVIVAL"
        chosen_steer_angle = turn_direction * 0.8
        status_label = f"P1 [STAGE 1/3]: EVADING BOX GAP ({closest_front_obstacle:.2f}m)"

    # Priority 1B: Trigger Tightened 3-Stage Box Bypass if Shoulders (<0.17m) or Bore-Sight (<0.12m) Hit a Box/Wall
    elif shoulder_blocked or boresight_blocked:
        turn_direction = 1.0 if left_clearance >= right_clearance else -1.0
        turn_timer = 14
        bypass_cruise_timer = 26
        bypass_reorient_timer = 22
        search_timer = 0
        target_lock_timer = 0
        left_speed, right_speed = compute_wheel_speeds(EVADE_LINEAR_VEL, turn_direction * EVADE_ANGULAR_VEL)
        active_tier = "P1_SURVIVAL"
        chosen_steer_angle = turn_direction * 0.8
        status_label = f"P1 [STAGE 1/3]: BOX DETECTED -> STARTING BYPASS ({closest_front_obstacle:.2f}m)"

    # Priority 1C (Stage 2 of Bypass): Drive Straight Past the Outer Corner of the Wooden Box
    elif bypass_cruise_timer > 0:
        side_nudge = 0.25 if np.min(right_shoulder_arc) < 0.22 else (-0.25 if np.min(left_shoulder_arc) < 0.22 else 0.0)
        left_speed, right_speed = compute_wheel_speeds(CRUISE_LINEAR_VEL, side_nudge)
        bypass_cruise_timer -= 1
        active_tier = "P1_SURVIVAL"
        chosen_steer_angle = side_nudge
        status_label = f"P1 [STAGE 2/3]: CLEARING BOX CORNER ({bypass_cruise_timer} steps)"

    # Priority 1D (Stage 3 of Bypass): Curve Around the Box Corner Back Toward Pinned Goal
    elif bypass_reorient_timer > 0:
        bypass_reorient_timer -= 1
        if goal_pinned:
            steer_w = float(np.clip(1.8 * rel_angle_to_goal, -1.35, 1.35))
            left_speed, right_speed = compute_wheel_speeds(0.08, steer_w)
            chosen_steer_angle = rel_angle_to_goal
        else:
            left_speed, right_speed = compute_wheel_speeds(0.08, -turn_direction * 1.10)
            chosen_steer_angle = -turn_direction * 0.6
        active_tier = "P2_MISSION"
        status_label = f"P2 [STAGE 3/3]: CURVING AROUND BOX TO PIN ({bypass_reorient_timer} steps)"

    # Priority 2B: Direct Visual Servoing (YOLO or Close-Range HSV Verified Lock)
    elif target_locked:
        search_timer = 0
        angular_w = float(np.clip(-KP_YAW * pixel_error, -1.40, 1.40))
        left_speed, right_speed = compute_wheel_speeds(APPROACH_LINEAR_VEL, angular_w)
        active_tier = "P2_MISSION"
        chosen_steer_angle = -(pixel_error / FRAME_CENTER_X) * (CAMERA_HFOV / 2.0)
        if hsv_verified_lock:
            lock_mode = "HSV+PIN LOCK"
        elif target_visible:
            lock_mode = "YOLO+MAP"
        else:
            lock_mode = "OBJECT PERMANENCE"
        status_label = f"P2 [{lock_mode}]: LOCKED ON TARGET (Dist: {target_metric_dist:.2f}m)"

    # Priority 2C: Map-Memory Navigation (With Phantom-Pin Watchdog)
    elif goal_pinned:
        if abs(rel_angle_to_goal) > 0.35:
            steer_w = float(np.clip(1.6 * rel_angle_to_goal, -1.25, 1.25))
            left_speed, right_speed = compute_wheel_speeds(0.05, steer_w)
        else:
            steer_w = float(np.clip(1.4 * rel_angle_to_goal, -1.0, 1.0))
            left_speed, right_speed = compute_wheel_speeds(APPROACH_LINEAR_VEL, steer_w)
        active_tier = "P2_MISSION"
        chosen_steer_angle = rel_angle_to_goal
        status_label = f"P2 [MAP MEMORY]: SEEKING PIN (Dist: {dist_to_pinned_goal:.2f}m | Watchdog: {phantom_pin_timer}/20)"

    # Priority 3: VLM Context-Aware Exploration
    else:
        search_timer += 1
        active_tier = "P3_EXPLORATION"
        if current_vlm_state == "DEAD_END":
            turn_dir = 1.0 if left_clearance >= right_clearance else -1.0
            left_speed, right_speed = compute_wheel_speeds(0.0, turn_dir * SWEEP_ANGULAR_VEL)
            status_label = "P3 [VLM: DEAD_END]: ESCAPE TURN"
        elif current_vlm_state == "DOORWAY":
            left_speed, right_speed = compute_wheel_speeds(CRUISE_LINEAR_VEL, 0.0)
            status_label = "P3 [VLM: DOORWAY]: THREADING PASSAGE"
        else:
            if search_timer < 150:
                left_speed, right_speed = compute_wheel_speeds(CRUISE_LINEAR_VEL, 0.0)
                status_label = "P3 [VLM: OPEN_ROOM]: CRUISING"
            elif search_timer < 195:
                left_speed, right_speed = compute_wheel_speeds(0.0, SWEEP_ANGULAR_VEL)
                status_label = "P3 [VLM: OPEN_ROOM]: SWEEPING CAMERA"
            else:
                search_timer = 0
                left_speed, right_speed = compute_wheel_speeds(CRUISE_LINEAR_VEL, 0.0)
                status_label = "P3 [VLM: OPEN_ROOM]: CRUISING"

    if not goal_latched:
        tier_counts[active_tier] += 1
        telemetry_rows.append([
            total_steps, active_tier, round(robot_x, 3), round(robot_y, 3),
            round(closest_front_obstacle, 3), round(target_metric_dist, 3),
            current_vlm_verify if goal_pinned else current_vlm_state, round(yolo_ms, 1)
        ])

    # E. Render Live Dual-Panel Dashboard (Vertical Stack: Camera on Top, Map on Bottom)
    if annotated_frame is not None:
        vlm_mode_tag = f"VLM-Verify:{current_vlm_verify}" if goal_pinned else f"VLM-Explore:{current_vlm_state}"
        vlm_hud = f"{vlm_mode_tag} [{current_vlm_ms:.0f}ms] | YOLO: {yolo_ms:.0f}ms | Pose: ({robot_x:.2f}, {robot_y:.2f})"
        cv2.putText(annotated_frame, status_label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
        cv2.putText(annotated_frame, vlm_hud, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 215, 0), 2)

        map_vis = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        map_vis[:, :] = (30, 30, 30)
        map_vis[occ_grid < 0.42] = (50, 85, 50)
        map_vis[occ_grid > 0.60] = (235, 235, 235)

        if len(trajectory_pts) > 1:
            cv2.polylines(map_vis, [np.array(trajectory_pts, dtype=np.int32)], False, (255, 180, 0), 2)

        if goal_pinned:
            gu, gv = world_to_map(goal_x, goal_y)
            cv2.circle(map_vis, (gu, gv), 8, (0, 0, 255), -1)
            cv2.circle(map_vis, (gu, gv), 13, (0, 255, 255), 2)
            pin_label = "TARGET PIN [VLM:YES]" if current_vlm_verify == "VERIFIED_YES" else "TARGET PIN"
            cv2.putText(map_vis, pin_label, (gu + 12, gv + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        steer_world_angle = robot_theta + chosen_steer_angle
        su, sv = world_to_map(robot_x + 0.45 * math.cos(steer_world_angle),
                              robot_y + 0.45 * math.sin(steer_world_angle))
        cv2.line(map_vis, (rob_u, rob_v), (su, sv), (255, 0, 255), 2)

        cv2.circle(map_vis, (rob_u, rob_v), 7, (255, 255, 0), -1)
        hu, hv = world_to_map(robot_x + 0.22 * math.cos(robot_theta),
                              robot_y + 0.22 * math.sin(robot_theta))
        cv2.arrowedLine(map_vis, (rob_u, rob_v), (hu, hv), (0, 255, 0), 2, tipLength=0.35)

        cv2.putText(map_vis, "LIVE 2D OCCUPANCY GRID MAP (v2.1 MEMORY)", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        cv2.putText(map_vis, f"Cleared Phantom Pins: {pins_cleared_count} | Red: Active Target Pin", (12, 468),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # --- VERTICAL STACK (Camera on top, Map on bottom) ---
        target_w = 460
        cam_h = int(target_w * (annotated_frame.shape[0] / annotated_frame.shape[1]))
        frame_disp = cv2.resize(annotated_frame, (target_w, cam_h))
        map_disp = cv2.resize(map_vis, (target_w, target_w))

        dashboard = np.vstack((frame_disp, map_disp))

        cv2.namedWindow("Robot AI Vision", cv2.WINDOW_NORMAL)
        cv2.imshow("Robot AI Vision", dashboard)
        cv2.waitKey(1)

    # F. Actuate Motors
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

# --- 6. CLEAN SHUTDOWN ON WEBOTS RESET ---
controller_running = False
cv2.destroyAllWindows()
sys.exit(0)