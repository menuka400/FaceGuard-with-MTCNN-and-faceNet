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
FaceNet takes an image of the person’s face as input and outputs a vector of 128 numbers which represent the most important features of a face. In machine learning, this vector is called embedding. Why embedding? Because all the important information from an image is embedded into this vector. Basically, FaceNet takes a person’s face and compresses it into a vector of 128 numbers. Ideally, embeddings of similar faces are also similar.

Mapping high-dimensional data (like images) into low-dimensional representations (embeddings) has become a fairly common practice in machine learning these days.

![image](https://github.com/user-attachments/assets/e3e4e3e8-616c-4c29-9ba5-fdd01405200a)


### 3. **Image Pyramid & Sliding Window in Face Detection**

- **Image Pyramid**: Image Pyramids are one of the most beautiful concept of image processing.Normally, we work with images with default resolution but many times we need to change the resolution (lower it) or resize the original image in that case image pyramids comes handy.The pyrUp() function increases the size to double of its original size and pyrDown() function decreases the size to half. If we keep the original image as a base image and go on applying pyrDown function on it and keep the images in a vertical stack, it will look like a pyramid. The same is true for upscaling the original image by pyrUp function.Once we scale down and if we rescale it to the original size, we lose some information and the resolution of the new image is much lower than the original one.
  
![image](https://github.com/user-attachments/assets/cd819829-2e15-42a6-b46e-8b75546ca4f9)

- **Sliding Window**: A sliding window is a rectangular region that shifts around the whole image(pixel-by-pixel) at each scale. Each time the window shifts, the window region is applied to the classifier and detects whether that region has Haar features of a face.

![sliding_window_example](https://github.com/user-attachments/assets/aaf098b9-c3b4-4e9b-8889-9c3d7e6f0281)


### 4. **Face Embeddings & Similarity Calculation**
Each registered face is represented as a vector. The system calculates similarity using **Euclidean distance**.

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

