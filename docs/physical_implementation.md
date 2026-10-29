# Proposed Physical Implementation & Simulation Limitations

## 1. Component Functionality in Physical Deployment
* **NVIDIA Jetson Orin Nano:** Executes the main `controller.py` loop at 30 Hz. It runs two parallel workloads: the main control thread (LiDAR raycasting, occupancy grid generation, and YOLOv8n inference on TensorRT) and an asynchronous background daemon thread that queries the local Ollama server running the **Moondream VLM**.
* **LDS-01 360° LiDAR:** Spins at 300 RPM to emit infrared laser pulses, returning a 360-point distance array every 200ms. This array feeds directly into Priority 1 (Collision Survival & 3-Stage Box Bypass) and builds the 2D spatial occupancy grid.
* **Wide-Angle USB Camera:** Captures real-time RGB frames. When YOLOv8 detects a potential target, the frame is passed to the VLM to semantically verify the target and classify room topology (`DEAD_END`, `DOORWAY`, `OPEN_ROOM`).
* **L298N Motor Driver & Geared DC Motors:** Converts the angular velocity (`left_speed`, `right_speed`) calculated by the Subsumption Architecture into physical wheel rotation via Pulse Width Modulation (PWM).

---

## 2. Hardware Implementation Plan (Simulation-to-Reality Migration)
Because our software architecture decouples sensor data ingestion from the core decision-making logic, migrating from Webots to physical hardware requires replacing only the hardware abstraction layer:

1. **Camera Input Replacement:**
   * *Simulation:* `camera.getImage()` via Webots API.
   * *Physical:* Replace with standard OpenCV `cv2.VideoCapture(0)` reading directly from the `/dev/video0` USB interface on Linux.
2. **LiDAR Stream Replacement:**
   * *Simulation:* `lidar.getRangeImage()` via Webots API.
   * *Physical:* Read serial packets from `/dev/ttyUSB0` using the `pyserial` or `hls_lfcd_lds_driver` library, normalizing the 360 distance floats into the exact same array format expected by our raycasting algorithm.
3. **Motor Actuation Replacement:**
   * *Simulation:* `left_motor.setVelocity()` and `right_motor.setVelocity()`.
   * *Physical:* Use the `Jetson.GPIO` library to map normalized velocity values (`-6.28` to `+6.28` rad/s) to PWM duty cycles (`0%` to `100%`) on Pins 32 and 33, while toggling directional H-Bridge pins.
4. **Odometry & Localization:**
   * Integrate wheel encoder ticks with an onboard MPU6050 IMU via an Extended Kalman Filter (EKF) to replace the simulated GPS/Compass heading calculations.

---

## 3. Simulation Limitations & Real-World Mitigation Strategies
While Webots validates our control logic, physical environments introduce real-world noise that the simulator abstracts away:

| Simulation Limitation | Real-World Physical Impact | Mitigation Plan in Physical Build |
| :--- | :--- | :--- |
| **Idealized Wheel Traction** | Webots assumes minimal slip. Real floors (tile, dust, carpet transitions) cause wheel slip, leading to odometry drift on the 2D occupancy grid. | Fuse wheel encoder data with an **MPU6050 6-Axis IMU** and apply scan-matching (ICP) on the LiDAR grid to correct pose drift. |
| **Zero Thermal / Power Throttling** | The simulator runs on desktop power. Running YOLOv8 + Moondream VLM simultaneously on edge hardware draws 15W–25W and generates high heat. | Quantize YOLOv8n to **FP16 TensorRT**, utilize active PWM heatsink cooling on the Jetson, and limit VLM queries to once every 3.5 seconds. |
| **Perfect Optical Lighting** | Simulated lighting is uniform. Real warehouses have harsh overhead glare, motion blur, and shadows that degrade YOLO and HSV color thresholds. | Enable hardware **Auto-Exposure / WDR** on the USB camera and apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in OpenCV before inference. |
| **Clean LiDAR Raycasting** | Webots raycasting hits solid geometry cleanly. In reality, glass doors, mirrors, and black matte surfaces scatter or absorb infrared laser pulses. | Implement a **statistical outlier removal filter** on raw LiDAR scans and fuse ultrasonic bumper sensors for low-profile or transparent obstacles. |
