import cv2
import numpy as np
from PIL import Image
import psycopg2
import os
from tabulate import tabulate
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

def connect_db():
    return psycopg2.connect("postgres://avnadmin:AVNS_F9QTOJet-vzalfnV2-T@pg-9c3cfe6-menuka-1.h.aivencloud.com:21721/defaultdb?sslmode=require")

def initialize_database():
    """Create the required table if it doesn't exist and handle migration if needed"""
    conn = connect_db()
    cur = conn.cursor()
    try:
        # First, check if we need to clear existing records due to embedding size change
        cur.execute("""
            SELECT embedding FROM face_records LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            existing_embedding = row[0]
            if len(existing_embedding) != 512:  # FaceNet embedding size
                print("Detected incompatible embedding format in database. Clearing existing records...")
                cur.execute("DROP TABLE face_records")
                conn.commit()
                print("Database cleared successfully.")

        # Create the table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_records (
                id VARCHAR(255) PRIMARY KEY,
                embedding FLOAT[] NOT NULL
            )
        """)
        conn.commit()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
    finally:
        cur.close()
        conn.close()

class FaceProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Initialize FaceNet model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.resnet = self.resnet.to(self.device)

    def get_face_embedding(self, img):
        # Convert to PIL Image if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        try:
            # Get face tensor
            face = self.mtcnn(img)
            
            if face is None:
                return None
            
            # Move to appropriate device
            face = face.to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.resnet(face.unsqueeze(0))
            
            # Return as numpy array
            return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing face: {str(e)}")
            return None

def capture_face(face_processor):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        
        # Detect face
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
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Normalize embeddings
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Calculate cosine similarity
    similarity = np.dot(emb1, emb2)
    return similarity

def register_face(face_processor):
    user_id = input("Enter ID number: ")
    
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM face_records WHERE id = %s", (user_id,))
        if cur.fetchone():
            print("Error: This ID is already registered!")
            return
        
        # Capture multiple face samples
        embeddings = []
        for i in range(3):
            print(f"\nCapturing face sample {i+1}/3...")
            img_path = capture_face(face_processor)
            if img_path:
                img = Image.open(img_path)
                embedding = face_processor.get_face_embedding(img)
                if embedding is not None:
                    embeddings.append(embedding)
                    print(f"Sample {i+1} captured successfully!")
                else:
                    print("Failed to detect face clearly. Please try again.")
                    continue
            else:
                print("Failed to capture face. Please try again.")
                continue
            
        if len(embeddings) == 0:
            print("Failed to capture any valid face samples. Please try again.")
            return
        
        # Use average embedding
        average_embedding = np.mean(embeddings, axis=0)
        
        cur.execute("INSERT INTO face_records (id, embedding) VALUES (%s, %s)",
                   (user_id, average_embedding.tolist()))
        conn.commit()
        print(f"New user registered successfully with ID: {user_id}!")
        
    except Exception as e:
        print(f"Error during registration: {str(e)}")
    finally:
        cur.close()
        conn.close()

def recognize_face(face_processor):
    img_path = capture_face(face_processor)
    if img_path:
        try:
            img = Image.open(img_path)
            current_embedding = face_processor.get_face_embedding(img)
            
            if current_embedding is None:
                print("Failed to detect face clearly. Please try again.")
                return
            
            conn = connect_db()
            cur = conn.cursor()
            
            cur.execute("SELECT id, embedding FROM face_records")
            rows = cur.fetchall()
            
            if not rows:
                print("No registered users in the database!")
                return
            
            # Enhanced matching logic
            similarities = []
            for row in rows:
                user_id, stored_embedding = row
                try:
                    similarity = calculate_similarity(current_embedding, stored_embedding)
                    similarities.append((user_id, similarity))
                except Exception as e:
                    print(f"Error comparing with user {user_id}: {str(e)}")
                    continue
            
            if not similarities:
                print("Could not compare with any stored faces. Please try again.")
                return
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Adaptive threshold
            top_similarity = similarities[0][1]
            threshold = 0.75  # Base threshold for FaceNet
            
            if top_similarity >= threshold:
                print(f"User recognized as ID: {similarities[0][0]}")
                print(f"Confidence: {top_similarity:.2%}")
                
                # Show top 3 matches
                print("\nTop 3 matches:")
                for user_id, sim in similarities[:min(3, len(similarities))]:
                    print(f"ID: {user_id}, Confidence: {sim:.2%}")
            else:
                print("Access denied: Face not recognized")
                print(f"Best match confidence: {top_similarity:.2%}")
        
        except Exception as e:
            print(f"Error during recognition: {str(e)}")
        finally:
            cur.close()
            conn.close()

def view_users():
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM face_records ORDER BY id")
        rows = cur.fetchall()
        
        if not rows:
            print("\nNo users registered in the database.")
            return
        
        table_data = [[i+1, row[0]] for i, row in enumerate(rows)]
        headers = ["No.", "User ID"]
        
        print("\nRegistered Users:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nTotal users registered: {len(rows)}")
        
    except Exception as e:
        print(f"Error viewing users: {str(e)}")
    finally:
        cur.close()
        conn.close()

def remove_user():
    view_users()
    
    user_id = input("\nEnter the ID of the user to remove (or press Enter to cancel): ")
    if not user_id:
        return
    
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM face_records WHERE id = %s", (user_id,))
        if not cur.fetchone():
            print(f"No user found with ID: {user_id}")
            return
        
        confirm = input(f"Are you sure you want to remove user {user_id}? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deletion cancelled.")
            return
        
        cur.execute("DELETE FROM face_records WHERE id = %s", (user_id,))
        conn.commit()
        print(f"User {user_id} has been removed successfully!")
        
    except Exception as e:
        print(f"Error removing user: {str(e)}")
    finally:
        cur.close()
        conn.close()

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
    
    # Initialize face processor
    face_processor = FaceProcessor()
    print("System initialized successfully!")
    
    while True:
        print("\nMain Menu:")
        print("1. Start Face Recognition")
        print("2. Create New User")
        print("3. View User Database and Edit")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            recognize_face(face_processor)
        elif choice == "2":
            register_face(face_processor)
        elif choice == "3":
            view_and_edit_database()
        elif choice == "4":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
