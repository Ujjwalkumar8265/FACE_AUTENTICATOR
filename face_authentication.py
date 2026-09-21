import os
import subprocess
import sys
import time
import cv2
import face_recognition
import numpy as np

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5  # Lower = stricter matching (0.6 is default, 0.4-0.5 reduces false positives)
FRAME_THROTTLE = 2  # Process every Nth frame for performance
MAX_ATTEMPTS_SECONDS = 15  # Timeout for authentication



# ACTION TO RUN AFTER SUCCESSFUL AUTHENTICATION

def run_protected_program(username):
    """Executes the desired program/code after successful face verification."""
    print(f"\n[SUCCESS] Welcome, {username}! Executing protected workflow...")


    print("----------------------------------------")
    print(f" Access Granted for: {username.upper()}")
    print(" Executing secure task...")
    print(" Task completed successfully.")
    print("----------------------------------------")



# FACE RECOGNITION AUTHENTICATOR

def load_known_faces():
    """Loads and encodes all images in the known_faces directory."""
    known_encodings = []
    known_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(
            f"[ERROR] '{KNOWN_FACES_DIR}' directory created. Please place face photos inside and re-run."
        )
        sys.exit(1)

    print("[INFO] Loading authorized faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                # Use filename (without extension) as the username
                name = os.path.splitext(filename)[0].capitalize()
                known_names.append(name)
                print(f" -> Loaded user: {name}")
            else:
                print(f" -> [WARNING] No face found in {filename}, skipping.")

    if not known_encodings:
        print("[ERROR] No valid face encodings found in 'known_faces/'. Exiting.")
        sys.exit(1)

    return known_encodings, known_names


def authenticate():
    """Captures webcam video feed and authenticates user against database."""
    known_encodings, known_names = load_known_faces()

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Could not access webcam.")
        return None

    print("\n[INFO] Starting webcam for authentication. Look directly into the camera.")
    print("Press 'q' to cancel.")

    start_time = time.time()
    authenticated_user = None
    frame_count = 0

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            frame_count += 1

            # Resize frame to 1/4 size for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Process face recognition every N frames to maintain high FPS
            if frame_count % FRAME_THROTTLE == 0:
                face_locations = face_recognition.face_locations(
                    rgb_small_frame
                )
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )

                for face_encoding in face_encodings:
                    # Compare face against authorized database
                    matches = face_recognition.compare_faces(
                        known_encodings, face_encoding, tolerance=TOLERANCE
                    )
                    face_distances = face_recognition.face_distance(
                        known_encodings, face_encoding
                    )

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            authenticated_user = known_names[best_match_index]
                            break

            # Display visual feedback on camera feed
            status_text = "Authenticating..."
            status_color = (0, 165, 255)  

            if authenticated_user:
                status_text = f"Access Granted: {authenticated_user}"
                status_color = (0, 255, 0)  

            cv2.putText(
                frame,
                status_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2,
            )
            cv2.imshow("Face Authentication System", frame)

            # Break loop on authentication success
            if authenticated_user:
                cv2.waitKey(1000) 
                break

            # Timeout check
            if time.time() - start_time > MAX_ATTEMPTS_SECONDS:
                print("[TIMEOUT] Authentication timed out. Access denied.")
                break

            # Manual exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[CANCELLED] Authentication cancelled by user.")
                break

    finally:
        video_capture.release()
        cv2.destroyAllWindows()

    return authenticated_user



# MAIN ENTRY POINT

if __name__ == "__main__":
    user = authenticate()

    if user:
        run_protected_program(user)
    else:
        print("\n[ACCESS DENIED] User verification failed.")
