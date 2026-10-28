# 🤖 Autonomous VLM-Guided Navigation Brain

An infrastructure-free, plug-and-play autonomous navigation controller for mobile robots. Built and simulated on a TurtleBot3 within Webots, this system fuses classic LiDAR mapping with a Dual-AI Vision Stack (YOLOv8 + Moondream VLM) to intelligently explore unmapped environments, evade complex obstacles, and visually hunt down targets without human intervention or pre-programmed routes.

## 🚀 Key Features

* **Three-Tier Subsumption Architecture:** Seamlessly arbitrates between Priority 1 (Collision Survival & Evasion), Priority 2 (Mission Targeting & Visual Servoing), and Priority 3 (VLM Context-Aware Exploration).
* **Dual-AI Vision Stack:** Combines high-speed bounding-box tracking (YOLOv8n) with an asynchronous local Vision-Language Model (Moondream) for semantic scene understanding and target verification.
* **Real-Time 2D Occupancy Grid Mapping:** Raycasts 360-degree LiDAR data to build a dynamic spatial memory, enabling the robot to navigate around obstacles and remember target pins out of frame.
* **Dynamic Obstacle Evasion:** Features a strict 3-stage box bypass maneuver to smoothly navigate around tight corners and narrow gaps.
* **"Kidnapped Target" Tracking:** Utilizes short-term object permanence and phantom-pin invalidation to dynamically track objects that are moved during the mission.
* **Close-Range HSV Fallback:** Secondary visual verification system to maintain target locks when physical proximity causes AI bounding-box failures.

## 🛠️ Tech Stack

* **Robotics Simulation:** Webots (TurtleBot3 Burger)
* **Language:** Python 3
* **Computer Vision:** OpenCV, Ultralytics YOLOv8n
* **Vision-Language Model (VLM):** Moondream (via local Ollama)
* **Telemetry & Data:** CSV, Math, NumPy

## ⚙️ Prerequisites

Before running the controller, ensure you have the following installed on your system:
1. [Webots](https://cyberbotics.com/) (R2023b or newer recommended)
2. [Python 3.8+](https://www.python.org/)
3. [Ollama](https://ollama.com/) (running locally)

## 📦 Installation & Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/yusufkhan210807-wq/autonomous-navigation-brain.git](https://github.com/yusufkhan210807-wq/autonomous-navigation-brain.git)
cd autonomous-navigation-brain
