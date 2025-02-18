# Face Detection and Recognition System

## Overview
This project implements a face detection and recognition system using **MTCNN** for face detection and **FaceNet** for face recognition. The system can:

- Detect faces in real-time using MTCNN.
- Extract facial embeddings using FaceNet.
- Store and manage recognized faces in a PostgreSQL database.
- Perform identity verification based on facial similarity.

## Features
- **Real-time face detection and recognition**
- **User registration and database management**
- **Face similarity comparison using cosine similarity**
- **Adaptive threshold for better recognition accuracy**

## Tech Stack
- **Python**
- **OpenCV**
- **MTCNN (Multi-Task Cascaded Convolutional Networks)**
- **FaceNet (InceptionResnetV1 for face embeddings)**
- **PostgreSQL**
- **Torch & torchvision**

---

## System Workflow

1. Capture a face from the webcam.
2. Use **MTCNN** to detect the face.
3. Extract a **512-dimensional embedding** using **FaceNet**.
4. Compare the extracted embedding with stored embeddings in the database.
5. If the similarity is above the threshold, the person is recognized; otherwise, access is denied.

### Flowchart

*(Flowchart will be provided separately.)*

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PostgreSQL
- Required Python libraries:

```bash
pip install opencv-python numpy pillow psycopg2 torch torchvision facenet-pytorch tabulate
```

### Database Setup
Modify `connect_db()` with your database credentials.
Run the script to initialize the database:

```bash
python face_recognition.py
```

---

## Usage

### Register a New Face
```bash
python face_recognition.py
```
1. Choose the **Create New User** option.
2. Enter a unique ID.
3. Capture three samples of your face.

### Recognize a Face
1. Choose the **Start Face Recognition** option.
2. The system will detect and identify the user.

---

## Core Concepts

### 1. **MTCNN (Multi-Task Cascaded Convolutional Networks)**
MTCNN is a three-stage deep learning-based face detector:
- **P-Net (Proposal Network)**: Generates candidate face regions.
- **R-Net (Refinement Network)**: Refines the detected regions.
- **O-Net (Output Network)**: Outputs precise bounding boxes and facial landmarks.

![image](https://github.com/user-attachments/assets/ef7255b5-f325-486b-a284-043b81f3a6ff)


### 2. **FaceNet**
FaceNet generates **face embeddings**, which are unique numerical representations of faces. It uses **InceptionResNetV1** to extract a **512-dimensional feature vector** for each face.

### 3. **Image Pyramid & Sliding Window in Face Detection**
- **Image Pyramid**: Rescales the image to different sizes to detect faces at different scales.
- **Sliding Window**: A fixed-size window moves over the image to detect objects.

### 4. **Face Embeddings & Similarity Calculation**
Each registered face is represented as a vector. The system calculates similarity using **cosine similarity** or **Euclidean distance**.

- **Euclidean Distance Formula:**
  \[ d(A, B) = \sqrt{\sum_{i=1}^{N} (A_i - B_i)^2} \]
- Lower distance = Higher similarity.

---

## Future Enhancements
- **Optimize for edge devices using quantization**
- **Enhance recognition under poor lighting conditions**
- **Implement multi-face recognition**

---

## License
This project is licensed under the MIT License.

