# Facial Recognition Access Control System

This repository contains the source code for a fully autonomous access control system developed for the Raspberry Pi 5. The system performs real-time face detection and identity verification locally, ensuring high performance on resource-constrained edge hardware without relying on external cloud services nor expensive equipment.

## Project Overview

The primary objective of this project was to balance computational efficiency with recognition accuracy. While advanced models like ArcFace and YuNet were evaluated during the research phase , the final system utilizes a HOG based (Histogram of Oriented Gradients) pipeline to achieve low-latency, real-time authentication on the Pi 5.

### Objectives Met:

* **Real-Time Computer Vision**: A pipeline capable of detecting and recognizing faces with minimal lag.

* **Edge Data Persistence**: Full integration with a local **MariaDB** database for user profiles, biometric encodings, and audit logs.

* **Administrative UI**: A web dashboard built with **Gradio** that removes the need for command-line interaction during daily operations.

* **Intelligent Enrollment**: Specialized logic to validate head poses and verify the presence of glasses during user registration.

## System Architecture

The software is divided into three Python files:

* `database_ops.py`: Manages all **CRUD** (Create, Read, Update, Delete) operations and ensures relational integrity between user data and access permissions.

* `app_gradio.py`: The main application driver handling the frontend UI, live camera streams via **Picamera2**, and the facial recognition logic.

* `test_fc_rec.py`: The facial recognition module that actually recognizes the users in the database.

### Database Schema

The MariaDB implementation consists of four linked tables:

* **People**: Stores identity details and local file paths to biometric encodings.

* **Doors**: Records physical access point locations.

* **Permissions**: A many-to-many mapping of users to specific doors.

* **Logs**: A chronological history of every access attempt with timestamps.

## Technical Features

### Multi-Pose Validation

To ensure reliability, the system mandates a 3-point capture process. It uses a jaw-to-nose Euclidean distance ratio to verify head orientation:

* **Frontal**: Ratio between −0.15 and 0.15.

* **Left Profile**: Ratio < −0.35.

* **Right Profile**: Ratio > 0.35.

### Sobel-Based Glasses Detection

The system prevents "empty" glasses registrations by analyzing texture density in the nose bridge **Region of Interest (ROI)**. It applies a **Sobel Operator** to calculate an edge score; a capture is only accepted if the edge density is at least **30% higher** than the frontal (no-glasses) reference image.

## Hardware Requirements

* **Raspberry Pi 5** (8GB recommended).

* **Raspberry Pi Camera Module 3**.

* **OS**: Raspberry Pi OS (64-bit).

## Installation Summary

1. **Clone the Repository:**

```bash
git clone https://github.com/PedroCF56/Facial-Recognition-Access-Control-System.git
```

2. **Install Dependencies:** Requires `opencv-python` (v4.8.1+), `dlib`, `numpy`, `face_recognition`, `gradio`, and `mariadb`.

3. **Database Setup:** Install MariaDB and execute the provided schema SQL.

4.  **Execution:** Execute `app_gradio.py` to open the dashboard and start the system.

---

**Note:** This repository is a "clean" version of the original codebase. The original repository contained sensitive information about project testers and has been replaced with this public version, which includes all essential system files while protecting participant privacy.
