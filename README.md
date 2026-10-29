# Hardware Specifications & Circuit Architecture

## 💻 Compute & Processing
* **NVIDIA Jetson Orin Nano (8GB):** Primary edge AI compute node. Selected for its Tensor Cores, enabling local execution of YOLOv8 and the Moondream VLM without cloud latency.
* **Arduino Mega 2560 (Optional/Sub-controller):** Handles low-level real-time tasks like motor encoder reading and PWM signal generation.

## 👁️ Sensors
* **LDS-01 360-degree 2D LiDAR:** Raycasts the surrounding environment to generate the real-time 2D occupancy grid for Priority 1 obstacle evasion.
* **1080p Wide-Angle USB Camera (FOV 120°):** Captures RGB visual data for the YOLOv8 object detection and Moondream scene classification models.

## ⚙️ Actuation & Power
* **L298N Motor Driver Dual H-Bridge:** Controls the voltage and direction of the two drive wheels.
* **12V DC Gear Motors (x2) with Hall Effect Encoders:** Provides physical locomotion and odometry feedback.
* **11.1V 3S LiPo Battery (5200mAh):** Primary power source.
* **LM2596 Step-Down Buck Converters (x2):** Regulates the 11.1V battery down to 5V (for the Jetson/Sensors) and 12V (for the Motor Driver).

## 🔌 System Architecture & Circuit Diagram
*(The following schematic is generated via Mermaid.js)*

```mermaid
graph TD
    Battery[11.1V 3S LiPo Battery] --> Buck1[5V Buck Converter]
    Battery --> Buck2[12V Buck Converter]
    
    Buck1 --> Jetson[NVIDIA Jetson Orin Nano]
    Buck1 --> LiDAR[LDS-01 360 LiDAR]
    
    Buck2 --> L298N[L298N Motor Driver]
    
    Jetson <-->|USB 3.0| Camera[1080p USB Camera]
    Jetson <-->|UART / USB| LiDAR
    
    Jetson -->|GPIO / PWM| L298N
    
    L298N -->|Power| MotorL[Left DC Motor]
    L298N -->|Power| MotorR[Right DC Motor]
    
    MotorL -->|Encoder Data| Jetson
    MotorR -->|Encoder Data| Jetson
