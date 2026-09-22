# Facial Geometry Pipeline API

Scale-invariant facial landmark processing and geometric measurement microservice built with **FastAPI**, **MediaPipe Face Mesh**, and **OpenCV**. Designed for reliable deployment via Docker and cross-platform compatibility (macOS & Windows).

---

## Key Features

* **3D Landmark Extraction:** Processes facial mesh landmarks using MediaPipe Face Mesh.
* **IPD Normalization:** Scales all Euclidean distances relative to Interpupillary Distance (IPD) to guarantee scale invariance regardless of camera distance or sensor resolution.
* **Production REST API:** Asynchronous FastAPI endpoints with automatic Swagger documentation (`/docs`) and health checks.
* **Containerized Deployment:** Multi-stage Docker setup with OS-level headless OpenCV dependencies.
* **Automated Testing:** Integration and unit test suite powered by `pytest` and `httpx`.

---

## Tech Stack

| Component | Technology | Version / Notes |
| :--- | :--- | :--- |
| **Framework** | FastAPI / Uvicorn | High-performance async web server |
| **CV Engine** | MediaPipe | `0.10.21` |
| **Image Processing** | OpenCV | `4.9.0.80` |
| **Numerical Engine**| NumPy | `1.26.4` |
| **Containerization**| Docker & Docker Compose | Base image: `python:3.12-slim` |
| **Testing** | Pytest & HTTPX |

---

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vlad-Skubskiy/facial_geometry_pipeline.git
   cd facial_geometry_
   ```
2. **Build & run yout container:**
   ```bash
   docker compose up --build
   ```
3. **Access the API:**
   `http://localhost:8000/docs`
