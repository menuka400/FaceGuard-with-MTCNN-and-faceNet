import cv2
import numpy as np
from PIL import Image
import os
import json
from tabulate import tabulate
import torch
from torchvision import transforms
import time
import mediapipe as mp

# JSON file path
JSON_DB_PATH = "face_records.json"

def initialize_database():
    """Initialize the JSON file if it doesn't exist"""
    if not os.path.exists(JSON_DB_PATH):
        with open(JSON_DB_PATH, 'w') as f:
            json.dump({}, f)
        print("JSON database initialized successfully!")
    else:
        with open(JSON_DB_PATH, 'r') as f:
            data = json.load(f)
        if data:
            first_embedding = next(iter(data.values()))
            if len(first_embedding) != 512:
                print("Detected incompatible embedding format in JSON. Clearing existing records...")
                with open(JSON_DB_PATH, 'w') as f:
                    json.dump({}, f)
                print("JSON database cleared successfully!")
        print("JSON database loaded successfully!")

class FaceProcessor:
    def __init__(self):
        # Check for CUDA GPU availability and set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU found, using CPU.")
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range model
            min_detection_confidence=0.5
        )
        
        # Load the facial recognition model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Pre-load registered faces
        self.registered_faces = self.load_registered_faces()
        
        # Initialize processing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def load_registered_faces(self):
        """Load all registered faces from JSON file"""
        try:
            with open(JSON_DB_PATH, 'r') as f:
                data = json.load(f)
            if self.device.type == 'cuda':
                return [(user_id, torch.tensor(embedding, device=self.device)) 
                        for user_id, embedding in data.items()]
            else:
                return [(user_id, np.array(embedding)) for user_id, embedding in data.items()]
        except Exception as e:
            print(f"Error loading registered faces from JSON: {str(e)}")
            return []

    def refresh_registered_faces(self):
        """Refresh the registered faces list from JSON"""
        self.registered_faces = self.load_registered_faces()

    def get_face_embedding(self, input_data):
        try:
            if isinstance(input_data, np.ndarray):
                # Convert BGR to RGB
                input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
                input_data = Image.fromarray(input_data)
            
            # Preprocess image
            face_tensor = self.preprocess(input_data).to(self.device)
            face_tensor = face_tensor.unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
            
            return embedding[0] if self.device.type == 'cuda' else embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing face: {str(e)}")
            return None

    def detect_faces(self, image):
        """Detect faces using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            faces = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = image.shape[:2]
                box = [
                    int(bbox.xmin * w),
                    int(bbox.ymin * h),
                    int((bbox.xmin + bbox.width) * w),
                    int((bbox.ymin + bbox.height) * h)
                ]
                confidence = detection.score[0]
                faces.append({'box': box, 'confidence': confidence})
            return faces
        return []

    def calculate_similarity_gpu(self, embedding1, embedding2):
        """Calculate cosine similarity between embeddings on GPU"""
        emb1_norm = embedding1 / torch.norm(embedding1)
        emb2_norm = embedding2 / torch.norm(embedding2)
        similarity = torch.dot(emb1_norm, emb2_norm)
        return similarity.item()

    def recognize_face_realtime(self, frame):
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            results = []
            for face in faces:
                box = face['box']
                confidence = face['confidence']
                
                if confidence < 0.9:
                    continue
                    
                # Add margin to face bounding box
                margin = 20
                box[0] = max(0, box[0] - margin)
                box[1] = max(0, box[1] - margin)
                box[2] = min(frame.shape[1], box[2] + margin)
                box[3] = min(frame.shape[0], box[3] + margin)
                
                # Extract face region
                face_region = frame[box[1]:box[3], box[0]:box[2]]
                embedding = self.get_face_embedding(face_region)
                
                if embedding is not None:
                    best_match = None
                    highest_similarity = 0
                    
                    for user_id, stored_embedding in self.registered_faces:
                        if self.device.type == 'cuda':
                            similarity = self.calculate_similarity_gpu(embedding, stored_embedding)
                        else:
                            similarity = calculate_similarity(embedding, stored_embedding)
                        
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match = user_id
                    
                    results.append({
                        'box': box,
                        'user_id': best_match if highest_similarity >= 0.75 else "Unknown",
                        'confidence': highest_similarity
                    })
            return results
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return []

def capture_face(face_processor):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        faces = face_processor.detect_faces(frame)
        
        for face in faces:
            box = face['box']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        cv2.imshow("Face Detection - Press 's' to Save, 'q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and faces:
            save_path = "stored-faces/new_user.jpg"
            cv2.imwrite(save_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            return save_path
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def calculate_similarity(embedding1, embedding2):
    """Fallback CPU implementation of similarity calculation"""
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1, emb2)

# Rest of the functions remain largely unchanged
def register_face(face_processor):
    user_id = input("Enter ID number: ")
    
    with open(JSON_DB_PATH, 'r') as f:
        data = json.load(f)
    
    if user_id in data:
        print("Error: This ID is already registered!")
        return
    
    embeddings = []
    for i in range(10):
        print(f"\nCapturing face sample {i+1}/10...")
        img_path = capture_face(face_processor)
        if img_path:
            img = cv2.imread(img_path)
            faces = face_processor.detect_faces(img)
            if faces:
                face = faces[0]  # Take first detected face
                face_region = img[face['box'][1]:face['box'][3], face['box'][0]:face['box'][2]]
                embedding = face_processor.get_face_embedding(face_region)
                if embedding is not None:
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    embeddings.append(embedding)
                    print(f"Sample {i+1} captured successfully!")
                else:
                    print("Failed to generate embedding. Please try again.")
            else:
                print("No face detected. Please try again.")
    
    if not embeddings:
        print("Failed to capture any valid face samples.")
        return
    
    average_embedding = np.mean(embeddings, axis=0).tolist()
    data[user_id] = average_embedding
    
    with open(JSON_DB_PATH, 'w') as f:
        json.dump(data, f)
    print(f"New user registered successfully with ID: {user_id}!")

def real_time_recognition(face_processor):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Real-time Face Recognition", cv2.WINDOW_NORMAL)
    
    fps_counter = 0
    fps = 0
    fps_start_time = time.time()
    frame_interval = 1.0 / 5.0
    last_frame_time = 0
    last_refresh = time.time()
    refresh_interval = 10
    
    while True:
        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            ret, frame = cap.read()
            if not ret:
                continue
            
            last_frame_time = current_time
            
            if current_time - last_refresh > refresh_interval:
                face_processor.refresh_registered_faces()
                last_refresh = current_time
            
            results = face_processor.recognize_face_realtime(frame)
            
            for result in results:
                box = result['box']
                user_id = result['user_id']
                confidence = result['confidence']
                color = (0, 255, 0) if user_id != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                text = f"{user_id} ({confidence:.2%})"
                cv2.putText(frame, text, (box[0], box[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if face_processor.device.type == 'cuda':
                gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
                cv2.putText(frame, gpu_info, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Real-time Face Recognition", frame)
            if face_processor.device.type == 'cuda':
                torch.cuda.synchronize()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if face_processor.device.type == 'cuda':
        torch.cuda.empty_cache()

def recognize_face(face_processor):
    img_path = capture_face(face_processor)
    if img_path:
        try:
            img = cv2.imread(img_path)
            faces = face_processor.detect_faces(img)
            if not faces:
                print("No face detected in the image!")
                return
            
            face = faces[0]  # Take first detected face
            face_region = img[face['box'][1]:face['box'][3], face['box'][0]:face['box'][2]]
            current_embedding = face_processor.get_face_embedding(face_region)
            
            if current_embedding is None:
                print("Failed to detect face clearly.")
                return
            
            if isinstance(current_embedding, torch.Tensor):
                current_embedding = current_embedding.cpu().numpy()
            
            with open(JSON_DB_PATH, 'r') as f:
                data = json.load(f)
            
            if not data:
                print("No registered users in the database!")
                return
            
            similarities = []
            for user_id, stored_embedding in data.items():
                similarity = calculate_similarity(current_embedding, stored_embedding)
                similarities.append((user_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarity = similarities[0][1]
            threshold = 0.75
            
            if top_similarity >= threshold:
                print(f"User recognized as ID: {similarities[0][0]}")
                print(f"Confidence: {top_similarity:.2%}")
                print("\nTop 3 matches:")
                for user_id, sim in similarities[:min(3, len(similarities))]:
                    print(f"ID: {user_id}, Confidence: {sim:.2%}")
            else:
                print("Access denied: Face not recognized")
                print(f"Best match confidence: {top_similarity:.2%}")
        except Exception as e:
            print(f"Error during recognition: {str(e)}")

def view_users():
    with open(JSON_DB_PATH, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("\nNo users registered in the database.")
        return
    
    table_data = [[i+1, user_id] for i, user_id in enumerate(sorted(data.keys()))]
    headers = ["No.", "User ID"]
    print("\nRegistered Users:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal users registered: {len(data)}")

def remove_user():
    view_users()
    user_id = input("\nEnter the ID of the user to remove (or press Enter to cancel): ")
    if not user_id:
        return
    
    with open(JSON_DB_PATH, 'r') as f:
        data = json.load(f)
    
    if user_id not in data:
        print(f"No user found with ID: {user_id}")
        return
    
    confirm = input(f"Are you sure you want to remove user {user_id}? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Deletion cancelled.")
        return
    
    del data[user_id]
    with open(JSON_DB_PATH, 'w') as f:
        json.dump(data, f)
    print(f"User {user_id} has been removed successfully!")

def view_and_edit_database():
    while True:
        print("\nDatabase Management Menu:")
        print("1. View all registered users")
        print("2. Remove a user")
        print("3. Return to main menu")
        
        choice = input("Enter your choice: ")
        if choice == "1":
            view_users()
        elif choice == "2":
            remove_user()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    print("Initializing face recognition system with GPU acceleration...")
    os.makedirs("stored-faces", exist_ok=True)
    initialize_database()
    
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")
    
    face_processor = FaceProcessor()
    print("System initialized successfully!")
    
    while True:
        print("\nMain Menu:")
        print("1. Start Face Recognition")
        print("2. Start Real-time Face Recognition")
        print("3. Create New User")
        print("4. View User Database and Edit")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        if choice == "1":
            recognize_face(face_processor)
        elif choice == "2":
            real_time_recognition(face_processor)
        elif choice == "3":
            register_face(face_processor)
            face_processor.refresh_registered_faces()
        elif choice == "4":
            view_and_edit_database()
            face_processor.refresh_registered_faces()
        elif choice == "5":
            print("Exiting program...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()