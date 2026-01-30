import cv2
import numpy as np
import pandas as pd
import os
import time
import threading
import warnings
from datetime import datetime
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')
print("Loading DeepFace...")
from deepface import DeepFace
print("DeepFace loaded\n")

# configuration
class Config:
    KNOWN_FACES_DIR = "Face/known_faces"
    ATTENDANCE_FILE = "attendance.csv"
    THRESHOLD = 0.40
    CHECK_INTERVAL = 30
    CAMERA_INDEX = 0

# Create directory if not exists
os.makedirs(Config.KNOWN_FACES_DIR, exist_ok=True)

#loading known faces
known_faces = {}  # name: embedding

def load_known_faces():
    """Load or reload known faces from directory"""
    global known_faces
    known_faces.clear()
    
    files = [f for f in os.listdir(Config.KNOWN_FACES_DIR) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Loading {len(files)} reference image(s)...")
    
    for file in files:
        try:
            name = os.path.splitext(file)[0].split('_')[0]
            path = os.path.join(Config.KNOWN_FACES_DIR, file)
            
            result = DeepFace.represent(
                img_path=path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )
            
            if result and len(result) > 0:
                known_faces[name] = np.array(result[0]['embedding'])
                print(f"  {name}")
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    return len(known_faces)

# Initial load
load_known_faces()
print(f"\n Ready with {len(known_faces)} person(s): {list(known_faces.keys())}\n")

# attendence logging
class AttendanceLogger:
    def __init__(self):
        self.file = Config.ATTENDANCE_FILE
        self.recent_marks = {}
        
        if not os.path.exists(self.file):
            df = pd.DataFrame(columns=['Name', 'Action', 'Date', 'Time', 'Confidence'])
            df.to_csv(self.file, index=False)

    def mark_attendance(self, name, confidence):
        current_time = time.time()
        
        if name in self.recent_marks:
            if current_time - self.recent_marks[name] < 10:
                return None
        
        self.recent_marks[name] = current_time
        
        try:
            df = pd.read_csv(self.file)
            today = datetime.now().strftime("%Y-%m-%d")
            user_today = df[(df['Name'] == name) & (df['Date'] == today)]
            
            if user_today.empty:
                action = "IN"
            else:
                last_action = user_today.iloc[-1]['Action']
                action = "OUT" if last_action == "IN" else "IN"
        except:
            action = "IN"
        
        now = datetime.now()
        new_row = pd.DataFrame([{
            'Name': name,
            'Action': action,
            'Date': now.strftime("%Y-%m-%d"),
            'Time': now.strftime("%H:%M:%S"),
            'Confidence': f"{confidence:.1f}%"
        }])
        
        df = pd.read_csv(self.file)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.file, index=False)
        
        return action

logger = AttendanceLogger()

# face recognisation threshold (accuracy)
current_match = None
current_confidence = 0
is_processing = False
last_recognition_time = 0

def recognize_face_bg(frame):
    """Background face recognition"""
    global current_match, current_confidence, is_processing, last_recognition_time
    
    if not known_faces:
        return
        
    is_processing = True
    
    try:
        temp_path = f"temp_frame_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(temp_path, frame)
        
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        
        if result and len(result) > 0:
            unknown_emb = np.array(result[0]['embedding'])
            
            best_name = None
            best_distance = float('inf')
            
            for name, known_emb in known_faces.items():
                distance = cosine(known_emb, unknown_emb)
                if distance < best_distance:
                    best_distance = distance
                    best_name = name
            
            if best_distance <= Config.THRESHOLD:
                current_match = best_name
                current_confidence = (1 - best_distance) * 100
                last_recognition_time = time.time()
                logger.mark_attendance(best_name, current_confidence)
            else:
                current_match = None
                current_confidence = 0
        else:
            current_match = None
            current_confidence = 0
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"Recognition error: {e}")
    
    is_processing = False

# adding new face 
def register_new_person(frame, name):
    """Register a new face from current frame"""
    try:
        # Check if face is detectable
        temp_path = "temp_register.jpg"
        cv2.imwrite(temp_path, frame)
        
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True  # Must detect face
        )
        
        if not result or len(result) == 0:
            return False, "No face detected! Please face the camera clearly."
        
        # Save permanently
        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(Config.KNOWN_FACES_DIR, filename)
        cv2.imwrite(filepath, frame)
        
        # Add to known_faces immediately (no restart needed)
        embedding = np.array(result[0]['embedding'])
        known_faces[name] = embedding
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return True, f"Successfully registered '{name}'!"
        
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

# real time camera loop
print("Starting Attendance System with Registration")
print("Controls:")
print("  [Q] Quit")
print("  [A] Add/Register new person")
print("  [R] Reload faces from disk")
print("  [L] Show today's attendance log")


cap = cv2.VideoCapture(Config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0
registration_mode = False
registration_message = ""
message_timer = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    display_frame = frame.copy()
    h, w = display_frame.shape[:2]
    
    # Run recognition
    if frame_counter % Config.CHECK_INTERVAL == 0 and not is_processing and not registration_mode:
        threading.Thread(target=recognize_face_bg, args=(frame.copy(),)).start()
    
    # Clear old match after 3 seconds
    if time.time() - last_recognition_time > 3:
        current_match = None
        current_confidence = 0
    
    # Header
    cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(display_frame, "DeepFace Attendance | [A]Add [R]Reload [L]Log [Q]Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Status area
    if current_match:
        cv2.rectangle(display_frame, (10, 50), (350, 130), (0, 100, 0), -1)
        cv2.putText(display_frame, f"✓ {current_match}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(display_frame, f"Confidence: {current_confidence:.1f}%", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(display_frame, (10, 50), (300, 100), (0, 0, 100), -1)
        status_text = "Scanning..." if is_processing else "No Match"
        cv2.putText(display_frame, status_text, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Show registration message
    if registration_message and time.time() - message_timer < 3:
        cv2.rectangle(display_frame, (50, h//2 - 40), (w-50, h//2 + 40), (255, 255, 0), -1)
        cv2.rectangle(display_frame, (50, h//2 - 40), (w-50, h//2 + 40), (0, 255, 255), 3)
        cv2.putText(display_frame, registration_message, (60, h//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Stats
    cv2.putText(display_frame, f"Registered: {len(known_faces)}", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Threshold: {Config.THRESHOLD}", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('DeepFace Attendance System', display_frame)
    
    # Key handling
    key = cv2.waitKey(10) & 0xFF
    
    if key == ord('q'):
        break
        
    elif key == ord('a'):
        # ADD NEW PERSON
        print("\n REGISTER NEW PERSON")
        
        # Get name from console (since OpenCV window can't take text input easily)
        name = input("Enter name for new person: ").strip()
        
        if name:
            if name in known_faces:
                confirm = input(f"'{name}' already exists. Overwrite? (y/n): ")
                if confirm.lower() != 'y':
                    registration_message = "Registration cancelled"
                    message_timer = time.time()
                    continue
            
            print(" Capturing in 3 seconds... Look at camera!")
            time.sleep(3)
            
            # Capture frame
            ret, capture_frame = cap.read()
            if ret:
                success, msg = register_new_person(capture_frame, name)
                registration_message = msg
                message_timer = time.time()
                print(f"   {msg}")
                
                # Show captured frame
                cv2.imshow(f"Captured: {name}", capture_frame)
                cv2.waitKey(2000)
                cv2.destroyWindow(f"Captured: {name}")
            else:
                registration_message = "Capture failed!"
                message_timer = time.time()
        else:
            registration_message = "Name cannot be empty!"
            message_timer = time.time()
    
    elif key == ord('r'):
        # RELOAD FACES
        count = load_known_faces()
        registration_message = f"Reloaded {count} faces"
        message_timer = time.time()
        print(f" Reloaded {count} faces from disk")
    
    elif key == ord('l'):
        # SHOW LOG
        try:
            df = pd.read_csv(Config.ATTENDANCE_FILE)
            today = datetime.now().strftime("%Y-%m-%d")
            today_df = df[df['Date'] == today]
            print(f"\n Today's Attendance ({today}):")
            if not today_df.empty:
                print(today_df.to_string(index=False))
                print(f"\nTotal entries: {len(today_df)}")
            else:
                print("No entries today")
        except Exception as e:
            print(f"Error reading log: {e}")

cap.release()
cv2.destroyAllWindows()

print("\n System shutdown")
print(f" Faces saved in: {os.path.abspath(Config.KNOWN_FACES_DIR)}")
print(f"Attendance log: {Config.ATTENDANCE_FILE}")
