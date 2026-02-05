"""
@file test_fc_rec.py
@brief Standalone script for testing facial recognition on the Raspberry Pi.
@details This script initializes the PiCamera, loads known face encodings from the database,
and performs real-time face recognition. 
"""

import cv2
import face_recognition
import pickle
import os
import time
import numpy as np
from picamera2 import Picamera2
import database_ops

## @brief Tolerance level for face comparison.
TOLERANCE = 0.4

def calculate_sobel_score(face_image, face_location):
    """
    @brief Calculates the Sobel edge magnitude on the nose bridge region.
    @details This is the same logic used in app_gradio.py's check_glasses_filters.
    
    @param face_image The full image in BGR format.
    @param face_location Tuple (top, right, bottom, left) of the face bounding box.
    
    @return The Sobel edge score (float), or 0.0 if calculation fails.
    """
    try:
        # Convert to RGB for face_recognition processing
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Get landmarks for this specific face
        landmarks_list = face_recognition.face_landmarks(rgb, [face_location])
        if not landmarks_list:
            return 0.0
        
        landmarks = landmarks_list[0]
        
        # Define Region of Interest (ROI) - Nose Bridge
        nose_bridge = landmarks['nose_bridge']
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        top_nose = nose_bridge[0]
        
        # Calculate ROI width based on the distance between eyes
        inner_left_eye_x = left_eye[3][0]
        inner_right_eye_x = right_eye[0][0]
        dist_eyes = inner_right_eye_x - inner_left_eye_x
        
        roi_w = int(dist_eyes * 0.4)  # 40% of eye distance
        roi_h = int(dist_eyes * 0.3)  # Small height to focus on the bridge
        
        # Define coordinates centered on the top of the nose
        x1 = top_nose[0] - roi_w
        x2 = top_nose[0] + roi_w
        y1 = top_nose[1] - roi_h
        y2 = top_nose[1] + roi_h
        
        # Ensure bounds
        x1, x2 = max(0, x1), min(face_image.shape[1], x2)
        y1, y2 = max(0, y1), min(face_image.shape[0], y2)
        
        nose_roi = face_image[y1:y2, x1:x2]
        if nose_roi.size == 0:
            return 0.0

        # Edge Analysis (Sobel)
        gray_roi = cv2.cvtColor(nose_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_roi, (3, 3), 0)  # Blur to reduce skin noise
        
        # Compute gradients in X and Y directions
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # The score is the mean magnitude of edges in the ROI
        edge_score = np.mean(magnitude)
        
        return edge_score

    except Exception as e:
        print(f"[ERROR] Sobel calculation failed: {e}")
        return 0.0

def load_known_faces():
    """
    @brief Loads face encodings and names from the database filesystem into memory.
    """
    print("Loading face database...")
    known_encodings = []
    known_names = []
    
    # 1. Fetch all people from DB
    people = database_ops.get_all_people()
    
    if not people:
        print("No people found in the database.")
        return [], []

    count = 0
    for person in people:
        name = person['name']
        person_id = person['person_id']
        folder_path = person['encodings_path']
        
        if not folder_path or not os.path.exists(folder_path):
            continue

        # 2. Load all .pkl files for this person
        for file in os.listdir(folder_path):
            if file.endswith(".pkl"):
                path = os.path.join(folder_path, file)
                try:
                    with open(path, 'rb') as f:
                        encoding = pickle.load(f)
                        known_encodings.append(encoding)
                        known_names.append(f"{name} (ID: {person_id})")
                        count += 1
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
    
    print(f"Successfully loaded {count} encodings.")
    return known_encodings, known_names

def main():
    # 1. Load Data
    known_encodings, known_names = load_known_faces()
    
    if not known_encodings:
        print("Cannot start without known encodings.")
        return

    # Configuration for Saving Photos 
    save_directory = "/home/pedro/projeto_final/projeto_final/testes_notion/second_tests"
    
    # Ensure directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"[INFO] Created directory: {save_directory}")

    # Ask user for the base filename
    base_filename = input("Enter the name for the snapshot file (without .jpg): ").strip()
    if not base_filename:
        base_filename = "snapshot" 

    # 2. Initialize Camera
    print("Starting PiCamera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  
    print("Camera ready.")
    print("Commands: [SPACE] to Save Photo | [Q] to Quit")

    try:
        while True:
            # 3. Capture Frame (RGB)
            rgb_frame = picam2.capture_array()
            
            # 4. Create BGR version for OpenCV drawing and saving
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # 5. Detect Faces
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # 6. Identify, Calculate Sobel, and Draw
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                
                # Recognition Logic
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                
                # Calculate Sobel Score
                sobel_score = calculate_sobel_score(bgr_frame, (top, right, bottom, left))
                
                # Print to console
                print(f"[INFO] Detected: {name} | Sobel Score: {sobel_score:.2f}")
                print("Door is Opened" if name != "Unknown" else "Access Denied")

                # Drawing Logic
                green_color = (0, 255, 0)
                yellow_color = (0, 255, 255)  # Yellow in BGR
                
                # Draw green rectangle around face
                cv2.rectangle(bgr_frame, (left, top), (right, bottom), green_color, 1)
                
                # Draw name (green text above the rectangle)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(bgr_frame, name, (left, top - 10), font, 0.6, green_color, 1)
                
                # Draw Sobel score (yellow text below the rectangle)
                sobel_text = f"Sobel: {sobel_score:.2f}"
                cv2.putText(bgr_frame, sobel_text, (left, bottom + 20), font, 0.6, yellow_color, 1)

            # 7. Show the window
            cv2.imshow("Recognition View (SPACE to Save / Q to Quit)", bgr_frame)
            
            # 8. Check for Key Presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q'):
                break
            
            # Save Photo (SPACE BAR)
            elif key == ord(' '):
                # Add timestamp to avoid overwriting files if multiple photos are taken
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                final_filename = f"{base_filename}_{timestamp}.jpg"
                full_path = os.path.join(save_directory, final_filename)
                
                # Save the image (OpenCV saves in BGR, which bgr_frame already is)
                cv2.imwrite(full_path, bgr_frame)
                cv2.destroyAllWindows()
                print(f"\n[SUCCESS] Photo saved to: {full_path}\n")
                break
                
    except KeyboardInterrupt:
        print("\nStopping by keyboard interrupt...")
    finally:
        print("Cleaning up...")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()