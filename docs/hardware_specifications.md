# Hardware Specifications & Circuit Architecture

## 1. Simulation Chassis vs. Physical Edge-AI Build
In our Webots simulation, we utilize the **TurtleBot3 Burger** chassis to validate our navigation algorithms. However, a stock physical TurtleBot3 relies on a standard Raspberry Pi, which lacks the GPU compute required to run both **YOLOv8n** and the **Moondream Vision-Language Model (VLM)** locally without severe latency.

Therefore, our physical implementation upgrades the compute node to an **NVIDIA Jetson Orin Nano** and uses a cost-effective custom DC motor drive train while retaining the exact same **LDS-01 360° LiDAR** sensor profile used in the simulation.

| Subsystem | Simulated Model (Webots) | Proposed Physical Hardware | Specifications & Purpose |
| :--- | :--- | :--- | :--- |
| **Primary Compute** | Host PC (Simulated) | **NVIDIA Jetson Orin Nano (8GB)** | 40 TOPS AI performance, 1024-core NVIDIA Ampere GPU. Runs YOLOv8n + Ollama (Moondream VLM) locally. |
| **Motor Controller** | Simulated OpenCR | **L298N Dual H-Bridge + Arduino Nano** | Translates PWM velocity commands from the Jetson into voltage control for differential drive motors. |
| **LiDAR Sensor** | simulated `LDS-01` | **Robotis LDS-01 (360° 2D Laser Scanner)** | 360° scan, 120mm–3500mm range, 1° angular resolution, 300 RPM (5 Hz). Builds the 2D occupancy grid. |
| **Vision Sensor** | Simulated RGB Camera | **IMX219 / 1080p Wide-Angle USB Camera** | 120° FOV, 30 FPS via USB 3.0. Feeds RGB frames to the YOLO and VLM threads. |
| **Actuators** | RotationalMotors (x2) | **12V DC Geared Motors (200 RPM) w/ Encoders** | High-torque differential drive locomotion with Hall-effect quadrature encoders for closed-loop odometry. |
| **Power Supply** | Unconstrained | **11.1V 3S LiPo Battery (5200mAh, 35C)** | Paired with dual LM2596 DC-DC Buck Converters (regulated to 5V/4A for Logic/Sensors and 12V for Motors). |

---

## 2. Circuit Diagram & Pin Connections

### Power & Data Flow Schematic
*(Rendered natively via GitHub Mermaid)*

```mermaid
flowchart TD
    subgraph Power_Distribution ["⚡ Power Distribution Unit"]
        BAT["11.1V 3S LiPo Battery (5200mAh)"]
        SW["Main Kill Switch"]
        BUCK5["LM2596 Buck Converter (5V / 4A Out)"]
        BUCK12["Power Rail (11.1V - 12V Motor Supply)"]
        BAT --> SW
        SW --> BUCK5
        SW --> BUCK12
    end

    subgraph Compute_And_Sensors ["🧠 Edge AI Compute & Perception"]
        JETSON["NVIDIA Jetson Orin Nano (8GB)"]
        CAM["1080p Wide-Angle USB Camera"]
        LIDAR["LDS-01 360° 2D LiDAR"]
        BUCK5 -->|5V DC Barrel/GPIO| JETSON
        BUCK5 -->|5V Power| LIDAR
        CAM -->|USB 3.0 Port| JETSON
        LIDAR -->|USB-to-UART Adapter| JETSON
    end

    subgraph Actuation_System ["⚙️ Differential Drive Actuation"]
        L298N["L298N Dual H-Bridge Driver"]
        MOT_L["Left 12V DC Gear Motor + Encoder"]
        MOT_R["Right 12V DC Gear Motor + Encoder"]
        BUCK12 -->|VMS / 12V Input| L298N
        JETSON -->|GPIO Pins 32, 33 (PWM) & 11, 13, 15, 16 (DIR)| L298N
        L298N -->|OUT1 / OUT2| MOT_L
        L298N -->|OUT3 / OUT4| MOT_R
        MOT_L -.->|Encoder A/B Feedback| JETSON
        MOT_R -.->|Encoder A/B Feedback| JETSON
    end
```

### Physical Pinout & Wiring Table
| Component Pin | Connects To | Interface / Signal Type | Function |
| :--- | :--- | :--- | :--- |
| **L298N ENA** | Jetson Pin 32 (GPIO07) | Hardware PWM | Left Motor Speed Control |
| **L298N IN1 / IN2** | Jetson Pin 11 / Pin 13 | Digital GPIO Out | Left Motor Direction (Forward/Reverse) |
| **L298N ENB** | Jetson Pin 33 (GPIO13) | Hardware PWM | Right Motor Speed Control |
| **L298N IN3 / IN4** | Jetson Pin 15 / Pin 16 | Digital GPIO Out | Right Motor Direction (Forward/Reverse) |
| **LDS-01 TX / RX** | Jetson USB Port (via USB2LDS) | UART Serial (230400 baud) | 360° LiDAR Distance Array Stream |
| **USB Camera** | Jetson USB 3.0 Type-A | USB Video Class (UVC) | 640x480 / 1080p RGB Frame Capture |
| **Common GND** | Battery GND + Jetson Pin 6 + L298N GND | Common Ground Bus | Prevents floating logic levels |
