import cv2
import numpy as np
from PIL import Image
import os
import json
from tabulate import tabulate
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import time

# JSON file path
JSON_DB_PATH = "face_records.json"

def initialize_database():
    """Initialize the JSON file if it doesn't exist"""
    if not os.path.exists(JSON_DB_PATH):
        with open(JSON_DB_PATH, 'w') as f:
            json.dump({}, f)  # Empty dictionary to start
        print("JSON database initialized successfully!")
    else:
        # Check if existing data is compatible
        with open(JSON_DB_PATH, 'r') as f:
            data = json.load(f)
        if data:
            first_embedding = next(iter(data.values()))
            if len(first_embedding) != 512:  # FaceNet embedding size
                print("Detected incompatible embedding format in JSON. Clearing existing records...")
                with open(JSON_DB_PATH, 'w') as f:
                    json.dump({}, f)
                print("JSON database cleared successfully!")
        print("JSON database loaded successfully!")

class FaceProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.registered_faces = self.load_registered_faces()

    def load_registered_faces(self):
        """Load all registered faces from JSON file"""
        try:
            with open(JSON_DB_PATH, 'r') as f:
                data = json.load(f)
            return [(user_id, np.array(embedding)) for user_id, embedding in data.items()]
        except Exception as e:
            print(f"Error loading registered faces from JSON: {str(e)}")
            return []

    def refresh_registered_faces(self):
        """Refresh the registered faces list from JSON"""
        self.registered_faces = self.load_registered_faces()

    def get_face_embedding(self, input_data):
        # [Unchanged from your original code]
        try:
            if isinstance(input_data, torch.Tensor):
                face_tensor = input_data.to(self.device)
            else:
                if isinstance(input_data, np.ndarray):
                    input_data = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
                face_tensor = self.mtcnn(input_data)
                if face_tensor is None:
                    return None
                face_tensor = face_tensor.to(self.device)
            
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.resnet(face_tensor)
            
            return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing face: {str(e)}")
            return None

    def recognize_face_realtime(self, frame):
        # [Unchanged from your original code except for registered_faces usage]
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame)
            boxes, probs = self.mtcnn.detect(img_pil)
            
            results = []
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob < 0.9:
                        continue
                    box = box.astype(int)
                    margin = 20
                    box[0] = max(0, box[0] - margin)
                    box[1] = max(0, box[1] - margin)
                    box[2] = min(frame.shape[1], box[2] + margin)
                    box[3] = min(frame.shape[0], box[3] + margin)
                    
                    face_region = rgb_frame[box[1]:box[3], box[0]:box[2]]
                    face_pil = Image.fromarray(face_region)
                    face_tensor = self.mtcnn(face_pil)
                    
                    if face_tensor is not None:
                        embedding = self.get_face_embedding(face_tensor)
                        if embedding is not None:
                            best_match = None
                            highest_similarity = 0
                            for user_id, stored_embedding in self.registered_faces:
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
    # [Unchanged from your original code]
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        boxes, _ = face_processor.mtcnn.detect(img_pil)
        if boxes is not None:
            for box in boxes:
                box = box.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Face Detection - Press 's' to Save, 'q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and boxes is not None:
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
    # [Unchanged from your original code]
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1, emb2)

def register_face(face_processor):
    user_id = input("Enter ID number: ")
    
    # Load current data
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
            img = Image.open(img_path)
            embedding = face_processor.get_face_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"Sample {i+1} captured successfully!")
            else:
                print("Failed to detect face clearly. Please try again.")
        else:
            print("Failed to capture face. Please try again.")
    
    if not embeddings:
        print("Failed to capture any valid face samples. Please try again.")
        return
    
    average_embedding = np.mean(embeddings, axis=0).tolist()
    data[user_id] = average_embedding
    
    with open(JSON_DB_PATH, 'w') as f:
        json.dump(data, f)
    print(f"New user registered successfully with ID: {user_id}!")

def real_time_recognition(face_processor):
    # [Unchanged except for refresh logic]
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Real-time Face Recognition", cv2.WINDOW_NORMAL)
    last_refresh = time.time()
    refresh_interval = 10
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if time.time() - last_refresh > refresh_interval:
            face_processor.refresh_registered_faces()
            last_refresh = time.time()
        
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
        
        cv2.imshow("Real-time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def recognize_face(face_processor):
    img_path = capture_face(face_processor)
    if img_path:
        try:
            img = Image.open(img_path)
            current_embedding = face_processor.get_face_embedding(img)
            if current_embedding is None:
                print("Failed to detect face clearly. Please try again.")
                return
            
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
    print("Initializing face recognition system...")
    os.makedirs("stored-faces", exist_ok=True)
    initialize_database()
    
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
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()