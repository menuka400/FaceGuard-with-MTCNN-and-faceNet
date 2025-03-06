import cv2
import numpy as np
import sqlite3
import pickle
import os
import time
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import logging
import tracemalloc
import psutil
import GPUtil
import pyttsx3
from datetime import datetime

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

model_dir = 'facenet-tensorflow-tensorflow2-default-v2'

my_face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
my_face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

logging.basicConfig(level=logging.ERROR)
DATABASE_FILE = 'face_database.db'

def init_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} (ID: {voice.id})")
        attractive_voice_index = 1
        if len(voices) > attractive_voice_index:
            engine.setProperty('voice', voices[attractive_voice_index].id)
        else:
            engine.setProperty('voice', voices[0].id)
        print(f"Selected voice: {engine.getProperty('voice')}")
        return engine
    except Exception as e:
        print(f"Failed to initialize TTS engine: {str(e)}")
        return None

def load_facenet():
    try:
        facenet = InceptionResnetV1(pretrained='vggface2').eval()
        torch.manual_seed(42)
        projection_layer = nn.Linear(512, 384, bias=False)
        nn.init.xavier_uniform_(projection_layer.weight)
        model = nn.Sequential(facenet, projection_layer).eval()
        if torch.cuda.is_available():
            model = model.cuda()
            print("FaceNet model with 384D projection loaded on GPU")
        else:
            print("FaceNet model with 384D projection loaded on CPU")
        return model
    except Exception as e:
        print(f"Error loading FaceNet model: {str(e)}")
        return None

def initialize_detector():
    try:
        return 1
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")
        return None

def get_face_embedding(model, face_img):
    try:
        if face_img.shape[0] < 160 or face_img.shape[1] < 160:
            face_img = cv2.resize(face_img, (160, 160))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        face_tensor = preprocess(face_img).unsqueeze(0)
        if torch.cuda.is_available():
            face_tensor = face_tensor.cuda()
        with torch.no_grad():
            embedding = model(face_tensor).cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    return cap

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return None
    frame = cv2.flip(frame, 1)
    return frame

def detect_and_extract_features(model, frame):
    if frame is None:
        return None, None, None, None
    tracemalloc.start()
    frame_with_box = frame.copy()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = my_face_detector.process(rgb_image)
    frame_height, frame_width, _ = rgb_image.shape
    
    if results.detections:
        for face in results.detections:
            face_rect = np.multiply(
                [face.location_data.relative_bounding_box.xmin,
                 face.location_data.relative_bounding_box.ymin,
                 face.location_data.relative_bounding_box.width,
                 face.location_data.relative_bounding_box.height],
                [frame_width, frame_height, frame_width, frame_height]
            ).astype(int)
            x, y, w, h = face_rect
            x_min, y_min = max(0, x), max(0, y)
            x_max, y_max = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            face_roi = frame[y_min:y_max, x_min:x_max]
            facenet_embedding = get_face_embedding(model, face_roi)
            tracemalloc.stop()
            cv2.rectangle(frame_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            return facenet_embedding, face_roi, (x_min, y_min, x_max, y_max), frame_with_box
    else:
        tracemalloc.stop()
        return None, None, None, frame_with_box

def get_head_pose(frame):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = my_face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            h, w, _ = frame.shape
            nose_x = nose_tip.x * w
            left_eye_x = left_eye.x * w
            right_eye_x = right_eye.x * w
            eye_midpoint_x = (left_eye_x + right_eye_x) / 2
            yaw = nose_x - eye_midpoint_x
            yaw_normalized = yaw * 100 / w
            if yaw_normalized > 0:
                return "Right", yaw_normalized
            elif yaw_normalized < 0:
                return "Left", yaw_normalized
            else:
                return "Frontal", yaw_normalized
    return "Unknown", 0

# New: SQLite database functions
def init_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            facenet_embedding BLOB,
            registration_date TEXT
        )
    ''')
    conn.commit()
    return conn

def load_database():
    try:
        conn = init_database()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, facenet_embedding, registration_date FROM users")
        rows = cursor.fetchall()
        database = {}
        for user_id, embedding_blob, registration_date in rows:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            print(f"Norm of {user_id}'s embedding: {norm:.3f}")
            if not np.isclose(norm, 1.0, atol=1e-5):
                embedding = embedding / norm
                print(f"Normalized {user_id}'s embedding (was not unit length)")
            database[user_id] = {
                'facenet_embedding': embedding,
                'registration_date': registration_date
            }
            print(f"First 5 values of {user_id}'s embedding: {embedding[:5]}")
        conn.close()
        return database
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return {}

def save_database(data):
    try:
        conn = init_database()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users")  # Clear existing data
        for user_id, user_data in data.items():
            embedding_blob = user_data['facenet_embedding'].tobytes()
            cursor.execute(
                "INSERT INTO users (user_id, facenet_embedding, registration_date) VALUES (?, ?, ?)",
                (user_id, embedding_blob, user_data['registration_date'])
            )
        conn.commit()
        conn.close()
        print(f"Database saved to {DATABASE_FILE}")
    except Exception as e:
        print(f"Error saving database: {str(e)}")

def register_user(user_name, cap=None, facenet_model=None, tts_engine=None):
    database = load_database()
    if facenet_model is None:
        facenet_model = load_facenet()
    if tts_engine is None:
        tts_engine = init_tts()
    if tts_engine is None:
        print("TTS not available. Proceeding without voice prompts.")
    
    if cap is None:
        cap = initialize_camera()
    facenet_embeddings = []
    phase_embeddings = []
    
    print(f"Registering user '{user_name}'.")
    print("Follow the voice instructions and arrows. Adjust your head as required.")
    cv2.namedWindow("Registration", cv2.WINDOW_NORMAL)
    
    frame = capture_frame(cap)
    if frame is not None:
        frame_with_box = detect_and_extract_features(facenet_model, frame)[3]
        cv2.imshow("Registration", frame_with_box)
        cv2.waitKey(10)
    
    phases = [
        ("Frontal Face", None, 10, "Do not move, I am going to take some images in one, two, three"),
        ("Prepare for Left", "left", 3, "Turn left"),
        ("Left Side", "left", 10, None),
        ("Prepare for Right", "right", 3, "Turn right"),
        ("Right Side", "right", 10, None),
    ]
    phase_idx = 0
    phase_start_time = time.time()
    collected_samples = 0
    tts_prompted = False
    
    while phase_idx < len(phases):
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(facenet_model, frame)
        head_pose, yaw_angle = get_head_pose(frame)
        
        phase_name, arrow_direction, duration, tts_message = phases[phase_idx]
        elapsed_time = time.time() - phase_start_time
        
        if elapsed_time < 1.0 and tts_message and tts_engine:
            print(f"Triggering TTS for {phase_name} at {elapsed_time:.2f}s")
            tts_engine.say(tts_message)
            tts_engine.runAndWait()
        
        remaining_time = max(0, duration - elapsed_time)
        cv2.putText(frame_with_box, f"Phase: {phase_name}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_box, f"Time left: {remaining_time:.1f}s", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_box, f"Head Pose: {head_pose} ({yaw_angle:.1f})", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if phase_name in ["Frontal Face", "Left Side", "Right Side"]:
            cv2.putText(frame_with_box, f"Samples: {collected_samples}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if arrow_direction == "left":
            cv2.arrowedLine(frame_with_box, (frame.shape[1] - 50, frame.shape[0] // 2), 
                           (frame.shape[1] - 100, frame.shape[0] // 2), (0, 255, 0), 2, tipLength=0.3)
        elif arrow_direction == "right":
            cv2.arrowedLine(frame_with_box, (50, frame.shape[0] // 2), 
                           (100, frame.shape[0] // 2), (0, 255, 0), 2, tipLength=0.3)
        
        pose_correct = False
        if phase_name in ["Frontal Face", "Left Side", "Right Side"]:
            if phase_name == "Frontal Face":
                if facenet_embedding is not None:
                    phase_embeddings.append(facenet_embedding)
                    collected_samples += 1
                    print(f"Sample collected in {phase_name}. Total samples: {collected_samples}")
                pose_correct = True
            elif (phase_name == "Left Side" and head_pose == "Left" and yaw_angle < 0) or \
                 (phase_name == "Right Side" and head_pose == "Right" and yaw_angle > 0):
                pose_correct = True
                tts_prompted = False
                if facenet_embedding is not None:
                    phase_embeddings.append(facenet_embedding)
                    collected_samples += 1
                    print(f"Sample collected in {phase_name}. Total samples: {collected_samples}")
            else:
                if phase_name in ["Left Side", "Right Side"]:
                    cv2.putText(frame_with_box, "Adjust head position! Restarting phase...", (10, 180), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not tts_prompted and tts_engine:
                        print("Triggering TTS: Adjust your head")
                        tts_engine.say("Until adjust your head the process not continue, restarting phase")
                        tts_engine.runAndWait()
                        tts_prompted = True
                    collected_samples -= len(phase_embeddings)
                    phase_embeddings = []
                    phase_start_time = time.time()
        
        cv2.imshow("Registration", frame_with_box)
        
        if (phase_name in ["Prepare for Left", "Prepare for Right"]) or pose_correct:
            if elapsed_time >= duration:
                print(f"Completed {phase_name} after {elapsed_time:.2f}s. Total samples so far: {collected_samples}")
                facenet_embeddings.extend(phase_embeddings)
                phase_embeddings = []
                phase_idx += 1
                phase_start_time = time.time()
                tts_prompted = False
                if phase_idx < len(phases):
                    next_phase_name = phases[phase_idx][0]
                    print(f"Now: {next_phase_name}")
                    if "Prepare" in next_phase_name:
                        print(f"Prepare to turn your face {phases[phase_idx][1]}...")
        
        key = cv2.waitKey(33)
        if key & 0xFF == ord('q'):
            print("Registration interrupted by user.")
            break
    
    if cap is None:
        cap.release()
    cv2.destroyAllWindows()
    
    facenet_embeddings.extend(phase_embeddings)
    
    if len(facenet_embeddings) < 30:
        print(f"Insufficient samples collected ({len(facenet_embeddings)}). Minimum 30 required. Registration failed.")
        return False
    
    mean_facenet_embedding = np.mean(facenet_embeddings, axis=0)
    
    database[user_name] = {
        'facenet_embedding': mean_facenet_embedding,
        'registration_date': datetime.now().isoformat()
    }
    save_database(database)
    
    print(f"User '{user_name}' registered successfully with {len(facenet_embeddings)} samples.")
    return True

# Modified: Automate registration when unknown_count reaches 10
def recognize_face():
    facenet_model = load_facenet()
    tts_engine = init_tts()
    
    cap = initialize_camera()
    
    stored_templates = load_database()
    if not stored_templates:
        print("No users registered in the database.")
    
    print("Starting continuous face recognition. Press 'q' to stop...")
    cv2.namedWindow("Recognition", cv2.WINDOW_NORMAL)
    
    facenet_threshold = 0.70
    last_recognized_name = None
    greeting_triggered = False
    no_face_count = 0
    last_state = None
    unknown_count = 0
    
    max_frames_per_second = 1
    target_frame_time = 1.0 / max_frames_per_second
    embeddings_batch = []
    frame_count = 0
    prev_time = time.time()
    
    current_label = "No face detected"
    current_color = (0, 0, 255)
    
    while True:
        cycle_start_time = time.time()
        
        for _ in range(max_frames_per_second):
            frame_start_time = time.time()
            
            frame = capture_frame(cap)
            if frame is None:
                continue
            
            facenet_embedding, face_roi, bbox, frame_with_box = detect_and_extract_features(facenet_model, frame)
            frame_count += 1
            
            if facenet_embedding is not None:
                embeddings_batch.append(facenet_embedding)
                norm = np.linalg.norm(facenet_embedding)
                print(f"Live embedding norm: {norm:.3f}, First 5 values: {facenet_embedding[:5]}")
                
                best_match, highest_confidence = None, -1
                for user_id, user_data in stored_templates.items():
                    stored_embedding = user_data['facenet_embedding']
                    similarity = np.dot(facenet_embedding, stored_embedding)
                    if similarity > highest_confidence:
                        highest_confidence = similarity
                        best_match = user_id
                    print(f"Similarity with {user_id}: {similarity:.3f}")
                
                if highest_confidence > facenet_threshold:
                    current_label = f"Recognized: {best_match} ({highest_confidence:.3f})"
                    current_color = (0, 255, 0)
                else:
                    current_label = f"Unknown face ({highest_confidence:.3f})"
                    current_color = (0, 255, 255)
            else:
                current_label = "No face detected"
                current_color = (0, 0, 255)
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            cv2.putText(frame_with_box, current_label, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
            cv2.putText(frame_with_box, f"FPS: {fps:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_with_box, f"Unknown count: {unknown_count}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Recognition", frame_with_box)
            
            elapsed_time = time.time() - frame_start_time
            sleep_time = target_frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        if embeddings_batch:
            avg_embedding = np.mean(embeddings_batch, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            norm = np.linalg.norm(avg_embedding)
            print(f"Batch avg embedding norm: {norm:.3f}, First 5 values: {avg_embedding[:5]}")
            best_match, highest_confidence = None, -1
            for user_id, user_data in stored_templates.items():
                stored_embedding = user_data['facenet_embedding']
                similarity = np.dot(avg_embedding, stored_embedding)
                if similarity > highest_confidence:
                    highest_confidence = similarity
                    best_match = user_id
                print(f"Batch similarity with {user_id}: {similarity:.3f}")
            
            if highest_confidence > facenet_threshold:
                unknown_count = 0
                if best_match != last_recognized_name:
                    last_recognized_name = best_match
                    greeting_triggered = False
                    no_face_count = 0
                    last_state = None
                    
                    print(f"Hello {best_match}, how's your day?")
                    if tts_engine:
                        tts_engine.say(f"Hello {best_match}, how's your day?")
                        tts_engine.runAndWait()
                    greeting_triggered = True
                    last_state = 'recognized'
                
                elif no_face_count >= 5 and last_state == 'no_face':
                    print(f"{best_match}, welcome back I miss you!")
                    if tts_engine:
                        tts_engine.say(f"{best_match}, welcome back I miss you!")
                        tts_engine.runAndWait()
                    no_face_count = 0
                    last_state = 'recognized'
            else:
                unknown_count += 1
                print(f"Unknown face detected. Count: {unknown_count}")
                
                # Automate registration if unknown_count reaches 10 and database is empty
                if unknown_count >= 10 and not stored_templates:
                    print("Unknown face detected 10 times and no users in database. Initiating registration...")
                    if tts_engine:
                        tts_engine.say("Unrecognized face detected. Please enter your name to register.")
                        tts_engine.runAndWait()
                    
                    user_name = input("Unrecognized face detected. Please enter your name to register: ").strip()
                    if user_name:
                        success = register_user(user_name, cap, facenet_model, tts_engine)
                        if success:
                            stored_templates = load_database()
                            print("Registration complete. Resuming recognition...")
                        else:
                            print("Registration failed. Resuming recognition...")
                    else:
                        print("No name provided. Resuming recognition...")
                    
                    unknown_count = 0
        else:
            unknown_count = 0
            if last_recognized_name is not None and greeting_triggered:
                no_face_count += 1
                if no_face_count == 5 and last_state == 'recognized':
                    print(f"{last_recognized_name}, where you go come back I need you!")
                    if tts_engine:
                        tts_engine.say(f"{last_recognized_name}, where you go come back I need you!")
                        tts_engine.runAndWait()
                    last_state = 'no_face'
        
        embeddings_batch = []
        frame_count = 0
        
        cycle_elapsed_time = time.time() - cycle_start_time
        if cycle_elapsed_time < 1.0:
            time.sleep(1.0 - cycle_elapsed_time)
        print(f"Processed {max_frames_per_second} frames in {time.time() - cycle_start_time:.2f} seconds")
    
    cap.release()
    cv2.destroyAllWindows()

def view_registered_users():
    database = load_database()
    
    if not database:
        print("No users registered.")
        return
    
    while True:
        print("\nRegistered Users:")
        for i, (user_id, user_data) in enumerate(database.items(), 1):
            print(f"{i}. {user_id} (registered: {user_data['registration_date']})")
        print("\nOptions:")
        print("1. View user details")
        print("2. Remove a user")
        print("3. Back to main menu")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            user_num = input("Enter user number to view details: ").strip()
            try:
                user_idx = int(user_num) - 1
                users = list(database.items())
                if 0 <= user_idx < len(users):
                    user_id, user_data = users[user_idx]
                    facenet_embedding = user_data['facenet_embedding']
                    print(f"\nDetails for '{user_id}':")
                    print(f"FaceNet embedding shape: {facenet_embedding.shape}")
                    print(f"First 5 values: {facenet_embedding[:5]}")
                else:
                    print("Invalid user number.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '2':
            user_num = input("Enter user number to remove: ").strip()
            try:
                user_idx = int(user_num) - 1
                users = list(database.items())
                if 0 <= user_idx < len(users):
                    user_id, _ = users[user_idx]
                    del database[user_id]
                    save_database(database)
                    print(f"User '{user_id}' removed.")
                else:
                    print("Invalid user number.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '3':
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def safe_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try again or contact support if the issue persists.")
            return None
    return wrapper

def run_diagnostics():
    print("\nRunning system diagnostics...")
    
    try:
        cap = initialize_camera()
        ret, frame = cap.read()
        if ret:
            print("✓ Camera initialization successful")
        else:
            print("✗ Camera not working properly")
            cap.release()
            return False
        cap.release()
    except Exception as e:
        print(f"✗ Camera error: {str(e)}")
        return False
    
    try:
        model = load_facenet()
        if model is not None:
            print("✓ FaceNet model loaded successfully")
        else:
            print("✗ FaceNet model failed to load")
            return False
    except Exception as e:
        print(f"✗ FaceNet model error: {str(e)}")
        return False
    
    try:
        detector = initialize_detector()
        if detector is not None:
            print("✓ Detector loaded successfully")
        else:
            print("✗ Detector failed to load")
            return False
    except Exception as e:
        print(f"✗ Detector error: {str(e)}")
        return False
    
    try:
        database = load_database()
        user_count = len(database)
        print(f"✓ Database check successful ({user_count} users registered)")
    except Exception as e:
        print(f"✗ Database error: {str(e)}")
        return False
    
    print("\nAll diagnostics passed successfully!")
    return True

def main_menu():
    print("\nInitializing system...")
    if not run_diagnostics():
        print("\nSystem diagnostics failed. Please check the errors above.")
        return
    
    while True:
        print("\n=== Lightweight Face Recognition System (384D Embeddings, SQLite Storage) ===")
        print("1. Start face recognition in real time")
        print("2. Register a new user")
        print("3. View registered users")
        print("4. Run system diagnostics")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            safe_operation(recognize_face)()
        
        elif choice == '2':
            user_name = input("Enter user name: ").strip()
            if user_name:
                safe_operation(register_user)(user_name)
            else:
                print("User name cannot be empty.")
        
        elif choice == '3':
            safe_operation(view_registered_users)()
        
        elif choice == '4':
            safe_operation(run_diagnostics)()
        
        elif choice == '5':
            print("\nShutting down system...")
            print("Thank you for using the Face Recognition System")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    try:
        if os.path.exists(DATABASE_FILE):
            print(f"Using existing database: {DATABASE_FILE}")
        else:
            print(f"No existing database found. A new one will be created at {DATABASE_FILE}")
        
        main_menu()
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Shutting down...")
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        print("Please restart the program or contact support.")
    finally:
        cv2.destroyAllWindows()