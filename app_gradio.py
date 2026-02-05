"""
@file app_gradio.py
@brief Main Gradio application for the Face Recognition Access System.
@details This file defines the entire user interface (UI), including all views,
data tables, forms, and event logic (callbacks) for managing people,
doors, and permissions.
"""
import os
import gradio as gr #pip install gradio
import database_ops
import re
import cv2 #pip install "opencv-python>4.8.1"
import pickle
import face_recognition #pip install face_recognition
import numpy as np #pip install "numpy"
from picamera2 import Picamera2 #pip install picamera2
import time
import threading

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
camera_lock = threading.Lock()
time.sleep(2)

## @brief Database fields for the People table (must match DB schema).
PEOPLE_DB_FIELDS = [
    "person_id", "name", "age", "cc_num",
    "contact_num", "email", "company_name", "images_path", "encodings_path"
]

## @brief Display headers for the People tables in the UI.
PEOPLE_DISPLAY_HEADERS = [
    "ID", "Name", "Age", "Citizen Card",
    "Contact", "Email", "Company Name", "Images Path", "Encodings Path"
]

## @brief Column width percentages for the People tables.
PEOPLE_COL_WIDTHS = ["5%", "15%", "7%", "15%", "15%", "25%", "25%", "40%", "40%"]

## @brief Data types for the People table columns for Gradio.
PEOPLE_TABLE_TYPES = ["number", "str", "number", "str", "str", "str", "str", "str", "str"]

## @brief Database fields for the Doors table.
DOOR_DB_FIELDS = ["door_id", "location", "description"]

## @brief Display headers for the Doors tables.
DOOR_DISPLAY_HEADERS = ["ID", "Location", "Description"]

## @brief Data types for the Doors table.
DOOR_TABLE_TYPES = ["number", "str", "str"]

## @brief Column width percentages for the Doors tables.
DOOR_COL_WIDTHS = ["10%", "45%", "45%"]

## @brief Database fields for the formatted Permissions table.
PERM_DB_FIELDS = ["permission_id", "person_name", "door_location"]

## @brief Display headers for the Permissions tables.
PERM_DISPLAY_HEADERS = ["ID", "Person Name", "Door Location"]

## @brief Data types for the Permissions table.
PERM_TABLE_TYPES = ["number", "str", "str"]

## @brief Column width percentages for the Permissions tables.
PERM_COL_WIDTHS = ["10%", "45%", "45%"]

## @brief Custom CSS string for styling the Gradio application.
custom_css = """
html, body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    min-height: 100vh !important;
}
.gradio-container {
    background: #f5f5f5 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    width: 100vw !important;
}
.page-container {
    min-height: 100vh !important;
    width: 100vw !important;
    position: relative !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding-bottom: 20vh !important;
}
.page-title {
    display: block !important;
    width: 100% !important;
    text-align: center !important;
    font-size: 48px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    color: #1a1a1a !important;
    margin: 48px 0 24px !important;
}
.content-wrapper {
    width: 820px !important;
    padding: 16px 0 !important;
    background: transparent !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 28px !important;
    align-items: center !important;
}
.button-box {
    width: 30vw !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 30px !important;
}
.button-box .gr-row {
    gap: 30px !important;
    width: 100% !important;
}
.menu-button {
    height: 150px !important;
    border: none !important;
    border-radius: 16px !important;
    font-size: 26px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
}
.menu-button:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12) !important;
}
.form-button {
    height: 50px !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.07) !important;
}
.form-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.10) !important;
}
.btn-green { background: #c7f0d0 !important; color: #0a3d18 !important; }
.btn-peach { background: #f8d6b7 !important; color: #6a3a0b !important; }
.btn-blue  { background: #cfe7ff !important; color: #0c3b6e !important; }
.btn-brown { background: #d6cfcf !important; color: #7d7d7d !important; }
.btn-red   { background: #f8d6d6 !important; color: #7d3d3d !important; }
"""

def on_click(table_name: str):
    """
    @brief Placeholder function for buttons with no action wired yet.
    @param table_name The name of the action to show in the info popup.
    """
    gr.Info(f"Action not wired yet: {table_name}")

def load_people_data():
    """
    @brief Fetches all people from the database and formats them for the DataFrames.
    @return A list of lists containing person data, ordered by PEOPLE_DB_FIELDS.
    """
    people = database_ops.get_all_people()
    return [[p.get(field) for field in PEOPLE_DB_FIELDS] for p in people]

def load_door_data():
    """
    @brief Fetches all doors from the database and formats them for the DataFrames.
    @return A list of lists containing door data, ordered by DOOR_DB_FIELDS.
    """
    doors = database_ops.get_all_doors()
    return [[d.get(field) for field in DOOR_DB_FIELDS] for d in doors]

def load_permissions_data():
    """
    @brief Fetches all formatted permissions (with names) from the database.
    @return A list of lists containing permission data, ordered by PERM_DB_FIELDS.
    """
    perms = database_ops.get_all_permissions()
    return [[p.get(field) for field in PERM_DB_FIELDS] for p in perms]

def add_person_and_refresh(name, age, cc_num, contact_num, email, company_name, 
                           enc_front, img_front, enc_left, img_left, enc_right, img_right,
                           enc_glasses, img_glasses):
    """
    @brief Callback for the 'Submit Person' button to register a new user.
    
    @details 
    1. Validates that all 3 mandatory face angles (Front, Left, Right) are captured.
    2. Validates all required text fields (Name, CC, Contact, Email).
    3. Validates data formats (Name must contain letters, Email regex, Age integer).
    4. Calls database_ops.add_person to create the DB record.
    5. Saves the face encodings (.pkl) and images (.jpg) to the person's folder.
    6. Clears all form inputs and capture states upon success.

    @param name Full name.
    @param age Age (optional).
    @param cc_num Citizen Card number.
    @param contact_num Contact number.
    @param email Email address.
    @param company_name Company name (optional).
    @param enc_front, img_front Encoding/Image for Frontal view.
    @param enc_left, img_left Encoding/Image for Left view.
    @param enc_right, img_right Encoding/Image for Right view.
    @param enc_glasses, img_glasses Encoding/Image for Glasses view (optional).

    @return A tuple of 17 updates (3 DataFrames, 6 Textboxes, 8 States).
    """
    
    # 1. Helper to return error state (17 items)
    def get_error_return():
        return (
            load_people_data(), load_people_data(), load_people_data(), # 3 Tables
            gr.update(), gr.update(), gr.update(), # 3 Fields (Name, Age, CC)
            gr.update(), gr.update(), gr.update(), # 3 Fields (Contact, Email, Company)
            gr.update(), gr.update(), # Front States
            gr.update(), gr.update(), # Left States
            gr.update(), gr.update(), # Right States
            gr.update(), gr.update()  # Glasses States
        )

    # 2. Validate Mandatory images
    if enc_front is None or enc_left is None or enc_right is None:
        gr.Warning("The first 3 angles (Front, Left, Right) are mandatory.")
        return get_error_return()

    # 3. Validate Required Text Fields
    if not all([name, cc_num, contact_num, email]):
        gr.Warning("Please fill in all required fields.")
        return get_error_return()

    # 4. Validate Formats
    if not any(c.isalpha() for c in name):
        gr.Warning("Name must contain at least one letter.")
        return get_error_return()

    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_regex, email):
        gr.Warning("Invalid email format.")
        return get_error_return()

    # 5. Validate Numbers
    age_int = None
    if age and age.strip():
        try:
            age_int = int(age)
        except ValueError:
            gr.Warning("Age must be a valid number.")
            return get_error_return()

    # 6. Save to DB
    company_final = company_name or None
    new_folder_path = database_ops.add_person(name, age_int, cc_num, contact_num, email, company_final)

    if new_folder_path:
        try:
            os.makedirs(new_folder_path, exist_ok=True)

            img_front_bgr = cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR)
            img_left_bgr = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
            img_right_bgr = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)

            with open(os.path.join(new_folder_path, "encoding_front.pkl"), 'wb') as f: pickle.dump(enc_front, f)
            cv2.imwrite(os.path.join(new_folder_path, "image_front.jpg"), img_front_bgr)
            
            with open(os.path.join(new_folder_path, "encoding_left.pkl"), 'wb') as f: pickle.dump(enc_left, f)
            cv2.imwrite(os.path.join(new_folder_path, "image_left.jpg"), img_left_bgr) 

            with open(os.path.join(new_folder_path, "encoding_right.pkl"), 'wb') as f: pickle.dump(enc_right, f)
            cv2.imwrite(os.path.join(new_folder_path, "image_right.jpg"), img_right_bgr) 
            
            if enc_glasses is not None and img_glasses is not None:
                img_glasses_bgr = cv2.cvtColor(img_glasses, cv2.COLOR_RGB2BGR) 
                with open(os.path.join(new_folder_path, "encoding_glasses.pkl"), 'wb') as f: 
                    pickle.dump(enc_glasses, f)
                cv2.imwrite(os.path.join(new_folder_path, "image_glasses.jpg"), img_glasses_bgr)
                
            gr.Info(f"Person '{name}' added successfully!")

        except Exception as e:
            gr.Error(f"Database record created, but failed to save files: {e}")
    else:
        gr.Error("Failed to add new person.")

    # 7. Clear Everything on success (Exactly 17 outputs)
    return (
        load_people_data(), load_people_data(), load_people_data(),
        gr.update(value=""), gr.update(value=""), gr.update(value=""),
        gr.update(value=""), gr.update(value=""), gr.update(value=""),
        gr.update(value=None), gr.update(value=None), 
        gr.update(value=None), gr.update(value=None), 
        gr.update(value=None), gr.update(value=None), 
        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), # snap_front
        gr.update(value=None), # snap_left
        gr.update(value=None), # snap_right
        gr.update(value=None)  # snap_glasses
    )

def get_face_direction_jaw_ratio(landmarks):
    """
    @brief Calculates face direction using the distance from Nose to Jaw (Ears).
    @details This method is more robust than using eyes because the jawline provides
    a wider reference width, making perspective changes easier to detect.
    
    @param landmarks A dictionary of facial landmarks (from face_recognition).
    
    @return A tuple (ratio, jaw_left, jaw_right, nose_tip):
            - ratio: A float indicating rotation (-0.5 to +0.5).
                     ~0.0 is Frontal.
                     Positive is Left (user's perspective).
                     Negative is Right (user's perspective).
            - jaw_left: (x, y) coordinates of the left jaw endpoint.
            - jaw_right: (x, y) coordinates of the right jaw endpoint.
            - nose_tip: (x, y) coordinates of the nose tip.
    """
    # Points 0 and 16 of the 'chin' landmark correspond to the area near the ears/temples
    jaw_left = np.array(landmarks['chin'][0])   # Left Endpoint
    jaw_right = np.array(landmarks['chin'][16]) # Right Endpoint
    
    # Nose tip (index 2 is the center of the tip)
    nose_tip = np.array(landmarks['nose_tip'][2])
    
    # Calculate Euclidean distances (more precise than just checking X coordinates)
    dist_to_left = np.linalg.norm(nose_tip - jaw_left)
    dist_to_right = np.linalg.norm(nose_tip - jaw_right)
    
    total_width = dist_to_left + dist_to_right
    
    if total_width == 0: return 0.0, jaw_left, jaw_right, nose_tip

    # Ratio calculation: (Right Distance - Left Distance) / Total Width
    ratio = (dist_to_right - dist_to_left) / total_width
    
    return ratio, jaw_left, jaw_right, nose_tip

def check_glasses_filters(face_image):
    """
    @brief Detects the presence of glasses by analyzing edge density on the nose bridge.
    
    @details This function isolates the region of interest (ROI) around the nose bridge
    using facial landmarks. It then applies Sobel edge detection to identify strong
    horizontal and vertical lines, which are characteristic of eyeglass frames or bridges.
    If the calculated edge score exceeds a predefined threshold, glasses are considered detected.
 
    @param face_image A numpy array representing the face image (BGR format).
    
    @return A tuple (bool, float):
            - True if glasses are detected, False otherwise.
            - The calculated edge density score (float).
    """
    try:
        # Convert to RGB for face_recognition processing
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # 1. Detect Face and Landmarks
        face_locs = face_recognition.face_locations(rgb, model="hog")
        if not face_locs: 
            print("No face detected for glasses check.")
            return False, 0.0
        
        landmarks = face_recognition.face_landmarks(rgb, face_locs)[0]
        
        # 2. Define Region of Interest (ROI) - Nose Bridge
        # We use the top of the nose and the inner corners of the eyes as reference points
        nose_bridge = landmarks['nose_bridge']
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        top_nose = nose_bridge[0]
        
        # Calculate ROI width based on the distance between eyes
        inner_left_eye_x = left_eye[3][0]
        inner_right_eye_x = right_eye[0][0]
        dist_eyes = inner_right_eye_x - inner_left_eye_x
        
        roi_w = int(dist_eyes * 0.4) # 40% of eye distance
        roi_h = int(dist_eyes * 0.3) # Small height to focus on the bridge
        
        # Define coordinates centered on the top of the nose
        x1 = top_nose[0] - roi_w
        x2 = top_nose[0] + roi_w
        y1 = top_nose[1] - roi_h
        y2 = top_nose[1] + roi_h
        
        nose_roi = face_image[y1:y2, x1:x2]
        if nose_roi.size == 0: 
            return False, 0.0

        # 3. Edge Analysis (Sobel)
        gray_roi = cv2.cvtColor(nose_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_roi, (3, 3), 0) # Blur to reduce skin noise
        
        # Compute gradients in X and Y directions
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # The score is the mean magnitude of edges in the ROI
        edge_score = np.mean(magnitude)
        
        print(f"Sobel Magnitude (Nose Bridge): {edge_score:.2f}")
        
        # Threshold determined through testing; values above this likely indicate glasses
        THRESHOLD = 30.0
        has_glasses = edge_score > THRESHOLD
        
        return has_glasses, edge_score

    except Exception as e:
        print(f"Error: Check Glasses failed: {e}")
        return True, 0.0 # Default to True on error to avoid blocking
    
def compare_sobel_scores(img_front_bgr, img_glasses_bgr):
    """
    @brief Compares the Sobel Edge Magnitude on the NOSE BRIDGE between two images.
    
    @details Uses a Region of Interest (ROI) focused on the nose bridge to detect
    glasses. It calculates a score for both the reference frontal image (without glasses)
    and the captured image (with glasses). It expects a significant percentage increase
    in edge density if glasses are present.

    @param img_front_bgr The reference frontal image (BGR format).
    @param img_glasses_bgr The captured image to test (BGR format).
    
    @return A tuple (bool, float, float):
            - True if glasses are detected (score increased by threshold), False otherwise.
            - The score of the frontal image.
            - The score of the glasses image.
    """

    if img_front_bgr is None or img_glasses_bgr is None:
        print("Error: Invalid images.")
        return False, 0.0, 0.0

    def get_nose_sobel_score(img):
        # 1. Convert to RGB and detect face
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb, model="hog")
        if not face_locs: return 0.0
        
        landmarks = face_recognition.face_landmarks(rgb, face_locs)[0]
        
        # 2. Define ROI (Nose Bridge)
        # Same logic as check_glasses_filters but used for extraction
        nose_bridge = landmarks['nose_bridge']
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        top_nose = nose_bridge[0]
        
        # Calculate width based on eye distance
        inner_left_eye_x = left_eye[3][0]
        inner_right_eye_x = right_eye[0][0]
        dist_eyes = inner_right_eye_x - inner_left_eye_x
        
        roi_w = int(dist_eyes * 0.4)
        roi_h = int(dist_eyes * 0.3)
        
        x1 = top_nose[0] - roi_w
        x2 = top_nose[0] + roi_w
        y1 = top_nose[1] - roi_h
        y2 = top_nose[1] + roi_h
        
        # Ensure image bounds
        x1, x2 = max(0, x1), min(img.shape[1], x2)
        y1, y2 = max(0, y1), min(img.shape[0], y2)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return 0.0
        
        # 3. Normalize Size (CRITICAL for comparison at different distances)
        # Resize nose bridge to a fixed width (e.g., 50px)
        target_w = 50
        h, w = crop.shape[:2]
        if w == 0: return 0.0
        scale = target_w / w
        new_h = int(h * scale)
        crop_resized = cv2.resize(crop, (target_w, new_h))
        
        # 4. Calculate Sobel Magnitude
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Return mean magnitude
        return np.mean(magnitude)

    # Calculate both scores
    score_front = get_nose_sobel_score(img_front_bgr)
    score_glasses = get_nose_sobel_score(img_glasses_bgr)

    print(f"1. Frontal Score (Without): {score_front:.2f}")
    print(f"2. Glasses Score (With):    {score_glasses:.2f}")
    
    diff = score_glasses - score_front
    print(f"-> Difference: {diff:+.2f}")
    
    is_valid = score_glasses > (score_front * 1.3)

    return is_valid, score_front, score_glasses

def handle_capture_for_add(target_angle, front_image_ref=None):
    """
    @brief Captures a face snapshot with real-time visual feedback and green bounding box.
    
    @details Opens an OpenCV window showing the camera feed. It detects faces,
    draws a green bounding box around them, and calculates the face direction ratio
    (nose to jaw) in real-time to guide the user.
    - Front/Glasses: Ratio must be between -0.15 and 0.15.
    - Left: Ratio must be less than -0.35.
    - Right: Ratio must be greater than 0.35.
    
    @param target_angle The target angle to capture ("front", "left", "right", "glasses").
    @param front_image_ref The reference frontal image (only required for "glasses" check).
    
    @return A tuple (encoding, snapshot_bgr, snapshot_bgr) if successful, or (None, None, None) on failure.
    """
    global picam2, camera_lock
    snapshot_rgb = None
    
    try:
        window_name = f"Press SPACE to Capture / Q to Quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            with camera_lock:
                frame_rgb = picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            face_locs = face_recognition.face_locations(frame_rgb, model="hog")
            
            # Draw GREEN bounding box (BGR: 0, 255, 0)
            for (top, right, bottom, left) in face_locs:
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

            # Visual Feedback Logic
            if face_locs:
                if len(face_locs) > 1:
                    status_text = f"ERROR: {len(face_locs)} faces detected. Only 1 allowed!"
                    cv2.putText(frame_bgr, status_text, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                    cv2.putText(frame_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    landmarks_list = face_recognition.face_landmarks(frame_rgb, face_locations=face_locs)
                    if landmarks_list:
                        # Calculate Ratio using Jawline
                        ratio, l_jaw, r_jaw, nose = get_face_direction_jaw_ratio(landmarks_list[0])
                        
                        # Draw lines (Pose visualization)
                        cv2.line(frame_bgr, tuple(nose.astype(int)), tuple(l_jaw.astype(int)), (0, 255, 255), 1)
                        cv2.line(frame_bgr, tuple(nose.astype(int)), tuple(r_jaw.astype(int)), (255, 0, 255), 1)

                        # Color Decision
                        is_good = False
                        
                        if target_angle == "front" or target_angle == "glasses":
                            if -0.15 <= ratio <= 0.15:
                                is_good = True
                                msg = "OK (Frontal)"
                            else:
                                msg = "Adjust Center"

                        elif target_angle == "left":
                            if ratio < -0.35: 
                                is_good = True
                                msg = "OK (Left)"
                            else:
                                msg = "Turn Left"

                        elif target_angle == "right":
                            if ratio > 0.35:
                                is_good = True
                                msg = "OK (Right)"
                            else:
                                msg = "Turn Right"

                        color = (0, 255, 0) if is_good else (0, 0, 255)
                        status_text = f"Ratio: {ratio:.2f} | {msg}"
                        cv2.putText(frame_bgr, status_text, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                        cv2.putText(frame_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                snapshot_rgb = frame_rgb
                break
            if key == ord('q'): 
                break
    finally:
        cv2.destroyAllWindows()

    if snapshot_rgb is None: return None, None, None

    # Final Validation
    try:
        face_locations = face_recognition.face_locations(snapshot_rgb, model="hog")
        if not face_locations:
            gr.Warning("No face detected.")
            return None, None, None

        snapshot_bgr_clean = cv2.cvtColor(snapshot_rgb, cv2.COLOR_RGB2BGR)
        landmarks = face_recognition.face_landmarks(snapshot_rgb, face_locations)[0]
        ratio, _, _, _ = get_face_direction_jaw_ratio(landmarks)

        if target_angle == "front":
            if not (-0.15 <= ratio <= 0.15):
                gr.Warning(f"Error: Look Straight ahead. (Ratio: {ratio:.2f})")
                return None, None, None
            '''has_glasses, score = check_glasses_filters(snapshot_bgr_clean)
            if has_glasses:
                gr.Warning(f"Frontal photo must be WITHOUT glasses. (Score: {score:.1f})")
                return None, None, None'''

        elif target_angle == "glasses":
            if not (-0.15 <= ratio <= 0.15):
                gr.Warning(f"Error: Look Straight ahead. (Ratio: {ratio:.2f})")
                return None, None, None
            if front_image_ref is None:
                gr.Warning("Capture the frontal photo without glasses first.")
                return None, None, None
            
            has_more, _, _ = compare_sobel_scores(front_image_ref, snapshot_bgr_clean)
            if not has_more:
                gr.Warning("Not enough glasses detail detected.")
                return None, None, None

        elif target_angle == "left":
            if not (ratio < -0.35): 
                gr.Warning(f"Error: Turn more to the LEFT. (Ratio: {ratio:.2f})")
                return None, None, None
                
        elif target_angle == "right":
            if not (ratio > 0.35):
                gr.Warning(f"Error: Turn more to the RIGHT. (Ratio: {ratio:.2f})")
                return None, None, None

        encoding = face_recognition.face_encodings(snapshot_rgb, face_locations)[0]
        gr.Info(f"Capture '{target_angle}' accepted!")
        return encoding, snapshot_bgr_clean, snapshot_bgr_clean

    except Exception as e:
        gr.Error(f"Error: {e}")
        return None, None, None

def handle_remove_person(person_identifier):
    """
    @brief Callback for the 'Confirm Removal' button (Person).
    @details Validates ID or Name and calls database_ops.remove_person.
    Can remove by Person ID (number) or by Name (text).
    @param person_identifier The ID or Name to remove, as a string.
    @return A tuple of 4 updates to refresh all 3 people DataFrames
            and clear the input field.
    """
    if not person_identifier or not person_identifier.strip():
        gr.Warning("Person ID or Name is required for removal.")
        data = load_people_data()
        return (data, data, data, gr.update())

    person_identifier = person_identifier.strip()
    
    try:
        person_id = int(person_identifier)
        status = database_ops.remove_person(person_id)
        identifier_type = "ID"
        identifier_value = person_id
        
    except ValueError:
        people = database_ops.get_all_people()
        matching_person = None
        
        for person in people:
            if person['name'].lower() == person_identifier.lower():
                matching_person = person
                break
        
        if matching_person:
            person_id = matching_person['person_id']
            status = database_ops.remove_person(person_id)
            identifier_type = "Name"
            identifier_value = person_identifier
        else:
            gr.Warning(f"Person with name '{person_identifier}' was not found.")
            data = load_people_data()
            return (data, data, data, gr.update())

    if status == "SUCCESS":
        gr.Info(f"Person ({identifier_type}: {identifier_value}) removed successfully.")
    elif status == "NOT_FOUND":
        gr.Warning(f"Person with {identifier_type} '{identifier_value}' was not found.")
    else:
        gr.Error("Failed to remove person. Check console for database errors.")

    data_refreshed = load_people_data()
    return (data_refreshed, data_refreshed, data_refreshed, gr.update(value=""))

def handle_load_person_with_photos(person_identifier):
    """
    @brief Carrega dados da pessoa E suas fotos existentes.
    @details Busca por ID ou Nome e carrega as 4 fotos (front, left, right, glasses).
    @param person_identifier ID ou Nome da pessoa.
    @return Dict com updates para todos os campos + imagens.
    """
    empty_return = {
        txt_edit_id_display: gr.update(value=""),
        txt_edit_name: gr.update(value=""), 
        txt_edit_age: gr.update(value=""),
        txt_edit_cc: gr.update(value=""),
        txt_edit_contact: gr.update(value=""),
        txt_edit_email: gr.update(value=""),
        txt_edit_company: gr.update(value=""),
        
        edit_snap_front: gr.update(value=None),
        edit_snap_left: gr.update(value=None),
        edit_snap_right: gr.update(value=None),
        edit_snap_glasses: gr.update(value=None)
    }
    
    if not person_identifier or not person_identifier.strip():
        gr.Warning("Person ID or Name is required.")
        return empty_return

    person_identifier = person_identifier.strip()
    person_data = None
    
    try:
        person_id = int(person_identifier)
        person_data = database_ops.get_person_by_id(person_id)
        search_type = "ID"
        search_value = person_id
    except ValueError:
        people = database_ops.get_all_people()
        for person in people:
            if person['name'].lower() == person_identifier.lower():
                person_data = person
                search_type = "Name"
                search_value = person_identifier
                break
        
        if not person_data:
            gr.Warning(f"Person with name '{person_identifier}' was not found.")
            return empty_return

    if not person_data:
        gr.Warning(f"Person with {search_type} '{search_value}' not found.")
        return empty_return
    
    folder_path = person_data.get('images_path') or person_data.get('encodings_path')
    
    img_front = None
    img_left = None
    img_right = None
    img_glasses = None
    
    if folder_path and os.path.exists(folder_path):
        try:
            front_path = os.path.join(folder_path, "image_front.jpg")
            if os.path.exists(front_path):
                img_front = cv2.imread(front_path)
                img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
            
            left_path = os.path.join(folder_path, "image_left.jpg")
            if os.path.exists(left_path):
                img_left = cv2.imread(left_path)
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            
            right_path = os.path.join(folder_path, "image_right.jpg")
            if os.path.exists(right_path):
                img_right = cv2.imread(right_path)
                img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
            
            glasses_path = os.path.join(folder_path, "image_glasses.jpg")
            if os.path.exists(glasses_path):
                img_glasses = cv2.imread(glasses_path)
                img_glasses = cv2.cvtColor(img_glasses, cv2.COLOR_BGR2RGB)
                
        except Exception as e:
            print(f"Error loading images: {e}")
    
    gr.Info(f"Data and photos for Person ({search_type}: {search_value}) loaded successfully.")
    
    return {
        txt_edit_id_display: gr.update(value=person_data.get('person_id')),
        txt_edit_name: gr.update(value=person_data.get('name')),
        txt_edit_age: gr.update(value=person_data.get('age')),
        txt_edit_cc: gr.update(value=person_data.get('cc_num')),
        txt_edit_contact: gr.update(value=person_data.get('contact_num')),
        txt_edit_email: gr.update(value=person_data.get('email')),
        txt_edit_company: gr.update(value=person_data.get('company_name')),
        
        edit_snap_front: gr.update(value=img_front),
        edit_snap_left: gr.update(value=img_left),
        edit_snap_right: gr.update(value=img_right),
        edit_snap_glasses: gr.update(value=img_glasses)
    }


def handle_update_person_with_photos(person_id_str, age, cc_num, contact_num, email, company_name,
                                     new_enc_front, new_img_front,
                                     new_enc_left, new_img_left,
                                     new_enc_right, new_img_right,
                                     new_enc_glasses, new_img_glasses):
    """
    @brief Callback para 'Save Changes' com atualização de fotos.
    @details Atualiza dados da pessoa E substitui fotos se novas foram capturadas.
    @return Tuple com updates das 3 DataFrames.
    """
    try:
        person_id = int(person_id_str)
    except ValueError:
        gr.Error("Invalid ID.")
        return (load_people_data(), load_people_data(), load_people_data())
        
    if not all([cc_num, contact_num, email]):
        gr.Warning("CC, Contact, and Email cannot be empty.")
        return (load_people_data(), load_people_data(), load_people_data())

    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_regex, email):
        gr.Warning("Invalid email format.")
        return (load_people_data(), load_people_data(), load_people_data())

    age_int = None
    if age and age.strip():
        try:
            age_int = int(age)
        except ValueError:
            gr.Warning("Age must be a valid integer.")
            return (load_people_data(), load_people_data(), load_people_data())

    company_final = company_name or None

    success = database_ops.update_person(
        person_id, age_int, cc_num, contact_num, email, company_final
    )
    
    if not success:
        gr.Error("Failed updating person data.")
        return (load_people_data(), load_people_data(), load_people_data())
    
    person_data = database_ops.get_person_by_id(person_id)
    if person_data:
        folder_path = person_data.get('images_path') or person_data.get('encodings_path')
        
        if folder_path and os.path.exists(folder_path):
            try:
                photos_updated = False
                
                # Refresh Front
                if new_enc_front is not None and new_img_front is not None:
                    img_bgr = cv2.cvtColor(new_img_front, cv2.COLOR_RGB2BGR)
                    with open(os.path.join(folder_path, "encoding_front.pkl"), 'wb') as f:
                        pickle.dump(new_enc_front, f)
                    cv2.imwrite(os.path.join(folder_path, "image_front.jpg"), img_bgr)
                    photos_updated = True
                
                # Refresh Left
                if new_enc_left is not None and new_img_left is not None:
                    img_bgr = cv2.cvtColor(new_img_left, cv2.COLOR_RGB2BGR)
                    with open(os.path.join(folder_path, "encoding_left.pkl"), 'wb') as f:
                        pickle.dump(new_enc_left, f)
                    cv2.imwrite(os.path.join(folder_path, "image_left.jpg"), img_bgr)
                    photos_updated = True
                
                # Refresh Right
                if new_enc_right is not None and new_img_right is not None:
                    img_bgr = cv2.cvtColor(new_img_right, cv2.COLOR_RGB2BGR)
                    with open(os.path.join(folder_path, "encoding_right.pkl"), 'wb') as f:
                        pickle.dump(new_enc_right, f)
                    cv2.imwrite(os.path.join(folder_path, "image_right.jpg"), img_bgr)
                    photos_updated = True
                
                # Refresh Glasses
                if new_enc_glasses is not None and new_img_glasses is not None:
                    img_bgr = cv2.cvtColor(new_img_glasses, cv2.COLOR_RGB2BGR)
                    with open(os.path.join(folder_path, "encoding_glasses.pkl"), 'wb') as f:
                        pickle.dump(new_enc_glasses, f)
                    cv2.imwrite(os.path.join(folder_path, "image_glasses.jpg"), img_bgr)
                    photos_updated = True
                
                if photos_updated:
                    gr.Info(f"Person {person_id} updated successfully with new photos!")
                else:
                    gr.Info(f"Person {person_id} data updated successfully!")
                    
            except Exception as e:
                gr.Warning(f"Data updated but failed to save some photos: {e}")
        else:
            gr.Info(f"Person {person_id} data updated successfully!")
    else:
        gr.Info(f"Person {person_id} updated successfully!")
    
    return (load_people_data(), load_people_data(), load_people_data())


def add_door_and_refresh(location, description):
    """
    @brief Callback para o botão 'Submit Door'.
    @details Validates input and calls database_ops.add_door.
    @param location Location name for the new door.
    @param description Optional description.
    @return A tuple of 5 updates to refresh all 3 door DataFrames
            and clear both input fields.
    """
    data = load_door_data() 
    
    if not location:
        gr.Warning("Location is Required.")
        # Returns 5 values (3 tables + 2 fields)
        return (data, data, data, gr.update(), gr.update())
    
    if not any(c.isalpha() for c in location):
        gr.Warning("Location must contain at least one letter.")
        # Returns 5 values (3 tables + 2 fields)
        return (data, data, data, gr.update(), gr.update())
    
    description_final = description or None
    success = database_ops.add_door(location, description_final)

    if success:
        gr.Info(f"Door '{location}' added successfully.")
    else:
        gr.Error("Failed to add new door.")

    # Refreshes the data
    data = load_door_data() 
    
    # Returns 5 values (3 tables + 2 fields)
    return (
        data, # 1. door_dataframe
        data, # 2. remove_door_dataframe
        data, # 3. edit_door_dataframe
        gr.update(value=""), # 4. txt_door_location
        gr.update(value="")  # 5. txt_door_description
    )

def handle_remove_door(door_id_str):
    """
    @brief Callback for the 'Remove Door' button.
    @details Validates ID and calls database_ops.remove_door.
    @param door_id_str The ID to remove.
    @return A tuple of 4 updates to refresh all 3 door DataFrames
            and clear the ID input field.
    """
    if not door_id_str:
        gr.Warning("Door ID is required")
        data = load_door_data()
        return (data, data, data, gr.update())
    
    try:
        door_id = int(door_id_str)
    except ValueError:
        gr.Warning("Door ID must be a number.")
        data = load_door_data()
        return (data, data, data, gr.update())
    
    status = database_ops.remove_door(door_id)

    if status == "SUCCESS":
        gr.Info(f"Door (ID: {door_id}) removed successfully.")
    elif status == "NOT_FOUND":
        gr.Warning(f"Door with ID {door_id} was not found.")
    else:
        gr.Error("Failed to remove door. Check console for database errors.")

    data = load_door_data()
    return (data, data, data, gr.update(value=""))

def handle_load_door(door_id_str):
    """
    @brief Callback for the 'Load Data' button (Edit Door).
    @details Fetches a single door's data and populates the edit form.
    @param door_id_str The ID to load.
    @return A dictionary of updates for the 3 form fields.
    """
    try:
        door_id = int(door_id_str)
    except ValueError:
        gr.Warning("ID Must be a Number")
        return {
            txt_edit_door_id_display: gr.update(value=""),
            txt_edit_door_location: gr.update(value=""),
            txt_edit_door_description: gr.update(value="")
        }
    
    door_data = database_ops.get_door_by_id(door_id)
    if door_data:
        gr.Info(f"Data for Door ID {door_id} loaded")
        return {
            txt_edit_door_id_display: gr.update(value=door_data.get('door_id')),
            txt_edit_door_location: gr.update(value=door_data.get('location')),
            txt_edit_door_description: gr.update(value=door_data.get('description'))
        }
    else:
        gr.Warning(f"Door with ID {door_id} not found. No data loaded.")
        return {
            txt_edit_door_id_display: gr.update(value=""),
            txt_edit_door_location: gr.update(value=""),
            txt_edit_door_description: gr.update(value="")
        }
    
def handle_update_door(door_id_str, location, description):
    """
    @brief Callback for the 'Save Changes' button (Edit Door).
    @details Validates and saves changes to an existing door.
    @param door_id_str The ID of the door being edited.
    @param location The new location name.
    @param description The new description (optional).
    @return A tuple of 3 updates to refresh all door DataFrames.
    """
    try:
        door_id = int(door_id_str)
    except ValueError:
        gr.Error("Invalid Door ID, cannot save")
        return (load_door_data(), load_door_data(), load_door_data())
    
    if not location:
        gr.Warning("Location is required.")
        return (load_door_data(), load_door_data(), load_door_data())

    if not any(c.isalpha() for c in location):
        gr.Warning("Location must contain at least one letter.")
        return (load_door_data(), load_door_data(), load_door_data())
    
    description_final = description or None

    success = database_ops.update_door(door_id, location, description_final)

    if success:
        gr.Info(f"Door {door_id} updated successfully") # 'added' was a typo
    else:
        gr.Error(f"Failed to update door with ID {door_id}")

    return (load_door_data(), load_door_data(), load_door_data())

def handle_add_permissions(person_id, door_id_list):
    """
    @brief Callback for 'Submit Permissions' (Add Permission screen).
    @details Adds a batch of new permissions. Does not remove existing ones.
    @param person_id The ID of the person (from dropdown).
    @param door_id_list A list of door IDs (from checkboxes).
    @return A tuple of 5 updates to refresh all 3 permission DataFrames
            and clear the form fields.
    """
    if not person_id or not door_id_list:
        gr.Warning("You must select one person and at least one door.")
        data_now = load_permissions_data()
        return (data_now, data_now, data_now, gr.update(), gr.update(value=[]))

    cont = database_ops.add_permissions(person_id, door_id_list)

    if cont > 0:
        gr.Info(f"Successfully added {cont} new permission(s).")
    else:
        gr.Info("No new permissions were added (they might already exist).")

    data_fresh = load_permissions_data()

    return (
        data_fresh,
        data_fresh,
        data_fresh,
        gr.update(value=None),
        gr.update(value=[])
    )

def handle_remove_permission(permission_id_str):
    """
    @brief Callback for 'Confirm Removal' (Remove Permission screen).
    @details Removes a single permission entry by its unique permission_id.
    @param permission_id_str The ID of the permission to remove.
    @return A tuple of 4 updates to refresh all 3 permission DataFrames
            and clear the ID input field.
    """
    data = load_permissions_data() 

    if not permission_id_str:
        gr.Warning("Permission ID is required for removal.")
        return (data, data, data, gr.update()) 
    try:
        permission_id = int(permission_id_str)
    except ValueError:
        gr.Warning("Permission ID must be a number.")
        return (data, data, data, gr.update())

    status = database_ops.remove_permission(permission_id)

    if status == "SUCCESS":
        gr.Info(f"Permission (ID: {permission_id}) removed successfully.")
    elif status == "NOT_FOUND":
        gr.Warning(f"Permission with ID {permission_id} was not found.")
    else: 
        gr.Error("Failed to remove permission. Check console for database errors.")

    data = load_permissions_data()
    return (data, data, data, gr.update(value=""))

def handle_load_person_permissions(person_id):
    """
    @brief Callback for 'Load Permissions' (Edit Permission screen).
    @details Fetches the list of door IDs a person currently has access to.
    @param person_id The ID of the person to load.
    @return An update for the CheckboxGroup, pre-selecting items.
    """
    if not person_id:
        gr.Warning("You must select a person first.")
        return gr.update(value=[]) 

    existing_door_ids = database_ops.get_permissions_for_person(person_id)
    
    if existing_door_ids:
        gr.Info(f"Loaded {len(existing_door_ids)} existing permissions.")
    else:
        gr.Info(f"Person has no existing permissions.")

    return gr.update(value=existing_door_ids)

def handle_update_person_permissions(person_id, new_door_id_list):
    """
    @brief Callback for 'Save Changes' (Edit Permission screen).
    @details Replaces all permissions for a person with a new set.
    @param person_id The ID of the person being edited.
    @param new_door_id_list The new list of door IDs to set.
    @return A tuple of 4 updates to refresh both relevant DataFrames
            and clear the form fields.
    """
    data = load_permissions_data()
    
    if not person_id:
        gr.Warning("No person was selected. Cannot save.")
        return (data, data, gr.update(value=None), gr.update(value=[]))

    if new_door_id_list is None:
        new_door_id_list = []
        
    success = database_ops.set_permissions_for_person(person_id, new_door_id_list)
    
    if success:
        gr.Info(f"Permissions for person {person_id} have been set.")
    else:
        gr.Error("Failed to set permissions. Check console.")
    
    data = load_permissions_data() 
    return (
        data, 
        data, 
        gr.update(value=None), 
        gr.update(value=[])  
    )

def show_page(page_name):
    updates = [gr.update(visible=False) for _ in range(14)] + [gr.update() for _ in range(29)]

    map_idx = {
        "menu": 0, "people": 1, "door": 2, "permissions": 3, "logs": 4,
        "add_person": 5, "remove_person": 6, "edit_person": 7, 
        "add_door": 8, "remove_door": 9, "edit_door": 10,
        "add_permission": 11, "remove_permission": 12, "edit_permission": 13
    }

    if page_name in map_idx:
        updates[map_idx[page_name]] = gr.update(visible=True)
    else:
        updates[0] = gr.update(visible=True)

    if page_name == "add_person":       updates[14] = load_people_data()
    if page_name == "remove_person":    updates[15] = load_people_data()
    if page_name == "edit_person":      updates[16] = load_people_data()
    
    if page_name == "add_door":         updates[17] = load_door_data()
    if page_name == "remove_door":      updates[18] = load_door_data()
    if page_name == "edit_door":        updates[19] = load_door_data()
    
    if page_name == "add_permission":   updates[20] = load_permissions_data()
    if page_name == "remove_permission": updates[21] = load_permissions_data()
    if page_name == "edit_permission":  updates[22] = load_permissions_data()

    if page_name == "add_permission" or page_name == "edit_permission":
        people = database_ops.get_all_people()
        doors = database_ops.get_all_doors()
        people_choices = [(p['name'], p['person_id']) for p in people] if people else []
        door_choices = [(d['location'], d['door_id']) for d in doors] if doors else []

        if page_name == "add_permission":
            updates[23] = gr.update(choices=people_choices, value=None)
            updates[24] = gr.update(choices=door_choices, value=[])
        
        if page_name == "edit_permission":
            updates[25] = gr.update(choices=people_choices, value=None)
            updates[26] = gr.update(choices=door_choices, value=[])

    if page_name != "add_person":
        for i in range(27, 35):
            updates[i] = gr.update(value=None)

    if page_name != "edit_person":
        for i in range(35, 43):
            updates[i] = gr.update(value=None)

    return tuple(updates)

## Gradio UI
with gr.Blocks(css=custom_css, title="Face Rec Menu") as demo:
    
    ## Main Menu View
    with gr.Column(visible=True, elem_classes=["page-container"]) as menu_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Face Rec Starting Menu</div>')
            with gr.Column(elem_classes=["button-box"]):
                with gr.Row(equal_height=True):
                    btn_menu_people = gr.Button("People Table", elem_classes=["menu-button", "btn-green"])
                    btn_menu_door = gr.Button("Door Table",   elem_classes=["menu-button", "btn-peach"])
                with gr.Row(equal_height=True):
                    btn_menu_perm = gr.Button("Permissions Table", elem_classes=["menu-button", "btn-blue"])
                    btn_menu_logs = gr.Button("Logs Table",        elem_classes=["menu-button", "btn-brown"])

    ## People Sub-Menu View
    with gr.Column(visible=False, elem_classes=["page-container"]) as people_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">People Table</div>')
            with gr.Column(elem_classes=["button-box"]):
                with gr.Row(equal_height=True):
                    btn_people_add  = gr.Button("Add Person",    elem_classes=["menu-button", "btn-green"])
                    btn_people_remove = gr.Button("Remove Person", elem_classes=["menu-button", "btn-red"])
                with gr.Row(equal_height=True):
                    btn_people_edit = gr.Button("Edit Person",   elem_classes=["menu-button", "btn-blue"])
                    btn_back_to_menu_people  = gr.Button("Back to Menu",  elem_classes=["menu-button", "btn-brown"])

    ## Add Person Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as add_person_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Add New Person</div>')
            
            gr.Markdown("### 1. Capture Face")
            gr.Markdown("Please capture the following photos in sequence. Make sure your face is clearly visible. Press SPACE to Capture or Q to Cancel.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**1. Frontal**")
                    snap_front = gr.Image(label="Front View", type="numpy", height=200, interactive=False)
                    btn_cap_front = gr.Button("📸 Capture Front", elem_classes=["form-button", "btn-blue"])
                    state_enc_front = gr.State(value=None)
                    state_img_front = gr.State(value=None)

                with gr.Column():
                    gr.Markdown("**2. Left Side**")
                    snap_left = gr.Image(label="Left View", type="numpy", height=200, interactive=False)
                    btn_cap_left = gr.Button("📸 Capture Left", elem_classes=["form-button", "btn-blue"])
                    state_enc_left = gr.State(value=None)
                    state_img_left = gr.State(value=None)

                with gr.Column():
                    gr.Markdown("**3. Right Side**")
                    snap_right = gr.Image(label="Right View", type="numpy", height=200, interactive=False)
                    btn_cap_right = gr.Button("📸 Capture Right", elem_classes=["form-button", "btn-blue"])
                    state_enc_right = gr.State(value=None)
                    state_img_right = gr.State(value=None)
                
                with gr.Column():
                    gr.Markdown("**4. Glasses (Optional)**")
                    snap_glasses = gr.Image(label="Front w/ Glasses", type="numpy", height=200, interactive=False)
                    btn_cap_glasses = gr.Button("📸 Capture Glasses", elem_classes=["form-button", "btn-blue"])
                    state_enc_glasses = gr.State(value=None)
                    state_img_glasses = gr.State(value=None)

            gr.Markdown("### 2. Fill Person Details")
            people_dataframe = gr.DataFrame(
                value=load_people_data,
                headers=PEOPLE_DISPLAY_HEADERS,
                datatype=PEOPLE_TABLE_TYPES,
                column_widths=PEOPLE_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )
            
            with gr.Column(elem_classes=["form-wrapper"]):
                txt_person_name = gr.Textbox(label="Name")
                with gr.Row():
                    txt_person_age = gr.Textbox(label="Age (Optional)", min_width=100)
                    txt_person_cc = gr.Textbox(label="Citizen Card Number", min_width=100)
                    txt_person_contact = gr.Textbox(label="Contact Number", min_width=100)
                txt_person_email = gr.Textbox(label="Email")
                txt_person_company = gr.Textbox(label="Company Name (Optional)")
                with gr.Row():
                    btn_people_add_submit = gr.Button(
                        "Submit Person", 
                        elem_classes=["form-button", "btn-green"], 
                        scale=2,
                        interactive=True
                    )
                    btn_back_to_people_menu = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)

    ## Remove Person Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as remove_person_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Remove Person</div>')
            gr.Markdown("### People Registered")
            remove_people_dataframe = gr.DataFrame(
                headers=PEOPLE_DISPLAY_HEADERS,
                datatype=PEOPLE_TABLE_TYPES,
                column_widths=PEOPLE_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )
            gr.Markdown("### Person to Remove")
            with gr.Column(elem_classes=["form-wrapper"]):
                gr.Markdown("**Type the ID (number) or Name of the person you wish to remove.**")
                txt_remove_id = gr.Textbox(
                    label="Person ID or Name", 
                    placeholder="",
                    min_width=100
                )
                with gr.Row():
                    btn_remove_submit = gr.Button("Confirm Removal", elem_classes=["form-button", "btn-red"], scale=2)
                    btn_back_to_people_menu_remove = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)


    ## Edit Person Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as edit_person_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Edit Person</div>')
            gr.Markdown("### People Registered")
            edit_people_dataframe = gr.DataFrame(
                headers=PEOPLE_DISPLAY_HEADERS,
                datatype=PEOPLE_TABLE_TYPES,
                column_widths=PEOPLE_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )

            gr.Markdown("### 1. Load Person to Edit")
            with gr.Column(elem_classes=["form-wrapper"]):
                gr.Markdown("**Type the ID (number) or Name of the person you wish to edit.**")
                txt_edit_id = gr.Textbox(
                    label="Person ID or Name to Load",
                    placeholder="",
                    min_width=100
                )
                btn_edit_load = gr.Button("Load Data", elem_classes=["form-button", "btn-blue"])

            gr.Markdown("### 2. Current Photos (Click 'Recapture' to replace)")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Frontal**")
                    edit_snap_front = gr.Image(label="Front View", type="numpy", height=200, interactive=False)
                    btn_edit_cap_front = gr.Button("📸 Recapture Front", elem_classes=["form-button", "btn-blue"])
                    edit_state_enc_front = gr.State(value=None)
                    edit_state_img_front = gr.State(value=None)

                with gr.Column():
                    gr.Markdown("**Left Side**")
                    edit_snap_left = gr.Image(label="Left View", type="numpy", height=200, interactive=False)
                    btn_edit_cap_left = gr.Button("📸 Recapture Left", elem_classes=["form-button", "btn-blue"])
                    edit_state_enc_left = gr.State(value=None)
                    edit_state_img_left = gr.State(value=None)

                with gr.Column():
                    gr.Markdown("**Right Side**")
                    edit_snap_right = gr.Image(label="Right View", type="numpy", height=200, interactive=False)
                    btn_edit_cap_right = gr.Button("📸 Recapture Right", elem_classes=["form-button", "btn-blue"])
                    edit_state_enc_right = gr.State(value=None)
                    edit_state_img_right = gr.State(value=None)
                
                with gr.Column():
                    gr.Markdown("**Glasses**")
                    edit_snap_glasses = gr.Image(label="Front w/ Glasses", type="numpy", height=200, interactive=False)
                    btn_edit_cap_glasses = gr.Button("📸 Recapture Glasses", elem_classes=["form-button", "btn-blue"])
                    edit_state_enc_glasses = gr.State(value=None)
                    edit_state_img_glasses = gr.State(value=None)

            gr.Markdown("### 3. Edit Personal Data and Save")
            with gr.Column(elem_classes=["form-wrapper"]) as edit_form_group:
                txt_edit_id_display = gr.Textbox(label="Editing Person ID", interactive=False)
                txt_edit_name = gr.Textbox(label="Name", interactive=False)
                with gr.Row():
                    txt_edit_age = gr.Textbox(label="Age (Optional)", min_width=100)
                    txt_edit_cc = gr.Textbox(label="Citizen Card Number", min_width=100)
                    txt_edit_contact = gr.Textbox(label="Contact Number", min_width=100)
                txt_edit_email = gr.Textbox(label="Email")
                txt_edit_company = gr.Textbox(label="Company Name (Optional)")
                with gr.Row():
                    btn_edit_submit = gr.Button("Save Changes", elem_classes=["form-button", "btn-green"], scale=2)
                    btn_back_to_people_menu_edit = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)

    ## Door Sub-Menu View
    with gr.Column(visible=False, elem_classes=["page-container"]) as door_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Door Table</div>')
            with gr.Column(elem_classes=["button-box"]):
                with gr.Row(equal_height=True):
                    btn_door_add    = gr.Button("Add Door",    elem_classes=["menu-button", "btn-green"])
                    btn_door_remove = gr.Button("Remove Door", elem_classes=["menu-button", "btn-red"])
                with gr.Row(equal_height=True):
                    btn_door_edit   = gr.Button("Edit Door",   elem_classes=["menu-button", "btn-blue"])
                    btn_back_to_menu_door  = gr.Button("Back to Menu",  elem_classes=["menu-button", "btn-brown"])

    ## Add Door Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as add_door_view:
         with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Add Door</div>')
            gr.Markdown("### Doors Registered")
            door_dataframe = gr.DataFrame(
                headers=DOOR_DISPLAY_HEADERS, 
                datatype=DOOR_TABLE_TYPES,
                column_widths=DOOR_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )
            gr.Markdown("### Add New Door Data")
            with gr.Column(elem_classes=["form-wrapper"]):
                txt_door_location = gr.Textbox(label="Location")
                txt_door_description = gr.Textbox(label="Description")
                with gr.Row():
                    btn_door_add_submit = gr.Button("Submit Door", elem_classes=["form-button", "btn-green"], scale=2)
                    btn_back_to_door_menu = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)

    ## Remove Door Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as remove_door_view:
         with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Remove Door</div>')
            gr.Markdown("### Doors Registered")
            remove_door_dataframe = gr.DataFrame(
                headers=DOOR_DISPLAY_HEADERS, 
                datatype=DOOR_TABLE_TYPES,
                column_widths=DOOR_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )
            gr.Markdown("### Remove Door")
            with gr.Column(elem_classes=["form-wrapper"]):
                gr.Markdown("Type the ID of the door you wish to remove.")
                txt_door_remove_id = gr.Textbox(label="ID")
                with gr.Row():
                    btn_door_remove_submit = gr.Button("Remove Door", elem_classes=["form-button", "btn-red"], scale=2)
                    btn_back_to_door_menu_remove = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)

    ## Edit Door Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as edit_door_view:
         with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Edit Door</div>')
            gr.Markdown("### Doors Registered")
            edit_door_dataframe = gr.DataFrame(
                headers=DOOR_DISPLAY_HEADERS, 
                datatype=DOOR_TABLE_TYPES,
                column_widths=DOOR_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )

            gr.Markdown("### 1. Load Door to Edit")
            with gr.Column(elem_classes=["form-wrapper"]):
                txt_edit_door_id = gr.Textbox(label="Door ID to Load", min_width=100)
                btn_edit_door_load = gr.Button("Load Data", elem_classes=["form-button", "btn-blue"])

            gr.Markdown("### 2. Edit Data and Save")
            with gr.Column(elem_classes=["form-wrapper"]):
                txt_edit_door_id_display = gr.Textbox(label="Editing Door ID", interactive=False)
                txt_edit_door_location = gr.Textbox(label="Location", interactive=True)
                txt_edit_door_description = gr.Textbox(label="Description")
                with gr.Row():
                    btn_edit_door_submit = gr.Button("Save Changes", elem_classes=["form-button", "btn-green"], scale=2)
                    btn_back_to_door_menu_edit = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)

    ## Permissions Sub-Menu View
    with gr.Column(visible=False, elem_classes=["page-container"]) as permissions_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Permissions Table</div>')
            with gr.Column(elem_classes=["button-box"]):
                with gr.Row(equal_height=True):
                    btn_perm_add    = gr.Button("Add Permission",    elem_classes=["menu-button", "btn-green"])
                    btn_perm_remove = gr.Button("Remove Permission", elem_classes=["menu-button", "btn-red"])
                with gr.Row(equal_height=True):
                    btn_perm_edit   = gr.Button("Edit Permission",   elem_classes=["menu-button", "btn-blue"])
                    btn_back_to_menu_perm = gr.Button("Back to Menu",  elem_classes=["menu-button", "btn-brown"])

    ## Add Permission Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as add_permission_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Add Permissions</div>')
            gr.Markdown("### Current Permissions")
            permissions_dataframe = gr.DataFrame(
                headers=PERM_DISPLAY_HEADERS,
                datatype=PERM_TABLE_TYPES,
                column_widths=PERM_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )

            gr.Markdown("### Add New Permissions")
            with gr.Column(elem_classes=["form-wrapper"]):
                person_dropdown = gr.Dropdown (
                    label = "Select Person",
                    choices = []
                )
                door_choices = gr.CheckboxGroup (
                    label = "Select Doors",
                    choices = []
                )
                with gr.Row():
                    btn_perm_add_submit = gr.Button("Submit Permissions", elem_classes=["form-button", "btn-green"], scale=2)
                    btn_back_to_perm_menu = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)

    ## Remove Permission Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as remove_permission_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Remove Permission</div>')
            gr.Markdown("### Current Permissions")
            remove_permissions_dataframe = gr.DataFrame(
                headers=PERM_DISPLAY_HEADERS,
                datatype=PERM_TABLE_TYPES,
                column_widths=PERM_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )

            gr.Markdown("### Permission to Remove")
            with gr.Column(elem_classes=["form-wrapper"]):
                gr.Markdown("Type the ID of the permission you wish to remove.")
                txt_perm_remove_id = gr.Textbox(label="Permission ID", min_width=100)
                with gr.Row():
                    btn_perm_remove_submit = gr.Button("Confirm Removal", elem_classes=["form-button", "btn-red"], scale=2)
                    btn_back_to_perm_menu_remove = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)
    
    ## Edit Permission Form View
    with gr.Column(visible=False, elem_classes=["page-container"]) as edit_permission_view:
         with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Edit Permissions</div>')
            gr.Markdown("### Current Permissions")
            edit_permissions_dataframe = gr.DataFrame(
                headers=PERM_DISPLAY_HEADERS,
                datatype=PERM_TABLE_TYPES,
                column_widths=PERM_COL_WIDTHS,
                elem_classes=["data-grid"],
                interactive=False,
                wrap=True
            )

            gr.Markdown("### 1. Load Person to Edit")
            with gr.Column(elem_classes=["form-wrapper"]):
                edit_person_dropdown = gr.Dropdown(
                    label="Select Person to Load",
                    choices=[] 
                )
                btn_perm_edit_load = gr.Button("Load Permissions", elem_classes=["form-button", "btn-blue"])

            gr.Markdown("### 2. Set Permissions")
            with gr.Column(elem_classes=["form-wrapper"]):
                edit_door_checkboxes = gr.CheckboxGroup(
                    label="Select all doors this person should have access to:",
                    choices=[] 
                )
                
                with gr.Row():
                    btn_perm_edit_submit = gr.Button("Save Changes", elem_classes=["form-button", "btn-green"], scale=2)
                    btn_back_to_perm_menu_edit = gr.Button("Return", elem_classes=["form-button", "btn-brown"], scale=1)

    ## Logs View
    with gr.Column(visible=False, elem_classes=["page-container"]) as logs_view:
        with gr.Column(elem_classes=["content-wrapper"]):
            gr.HTML('<div class="page-title">Logs Table</div>')
            with gr.Column(elem_classes=["button-box"]):
                with gr.Row(equal_height=True):
                    btn_logs_see = gr.Button("See Logs", elem_classes=["menu-button", "btn-blue"])
                    btn_back_to_menu_logs = gr.Button("Back to Menu", elem_classes=["menu-button", "btn-brown"])

    ## @brief List of all components that can be updated by show_page or handlers.
    # @details The order here MUST match the order of the tuple returned by show_page.
    all_outputs = [
        # 14 Views
        menu_view, people_view, door_view, permissions_view, logs_view,
        add_person_view, remove_person_view, edit_person_view, 
        add_door_view, remove_door_view, edit_door_view, 
        add_permission_view, remove_permission_view, edit_permission_view,

        # 9 Dataframes 
        people_dataframe, remove_people_dataframe, edit_people_dataframe, 
        door_dataframe, remove_door_dataframe, edit_door_dataframe,
        permissions_dataframe, remove_permissions_dataframe, edit_permissions_dataframe,

        # 4 Form Fields
        person_dropdown, door_choices,
        edit_person_dropdown, edit_door_checkboxes,

        state_enc_front, state_img_front,
        state_enc_left, state_img_left,
        state_enc_right, state_img_right,
        state_enc_glasses, state_img_glasses,

        edit_state_enc_front, edit_state_img_front,
        edit_state_enc_left, edit_state_img_left,
        edit_state_enc_right, edit_state_img_right,
        edit_state_enc_glasses, edit_state_img_glasses 
    ]

    ## Main Menu Navigation 
    btn_menu_people.click(lambda: show_page("people"), None, all_outputs)
    btn_menu_door.click(lambda: show_page("door"), None, all_outputs)
    btn_menu_perm.click(lambda: show_page("permissions"), None, all_outputs)
    btn_menu_logs.click(lambda: show_page("logs"), None, all_outputs)

    ## People Sub-Menu & Forms 
    btn_people_add.click(lambda: show_page("add_person"), None, all_outputs)
    btn_people_remove.click(lambda: show_page("remove_person"), None, all_outputs)
    btn_people_edit.click(lambda: show_page("edit_person"), None, all_outputs)
    btn_back_to_menu_people.click(lambda: show_page("menu"), None, all_outputs)

    btn_cap_front.click(
        fn=handle_capture_for_add,
        inputs=[gr.State("front"), state_img_front], 
        outputs=[state_enc_front, state_img_front, snap_front]
    )
    
    btn_cap_left.click(
        fn=handle_capture_for_add,
        inputs=[gr.State("left"), state_img_front],  
        outputs=[state_enc_left, state_img_left, snap_left]
    )
    
    btn_cap_right.click(
        fn=handle_capture_for_add,
        inputs=[gr.State("right"), state_img_front], 
        outputs=[state_enc_right, state_img_right, snap_right]
    )

    btn_cap_glasses.click(
        fn=handle_capture_for_add,
        inputs=[gr.State("glasses"), state_img_front], 
        outputs=[state_enc_glasses, state_img_glasses, snap_glasses]
    )

    inputs_add_person = [
        txt_person_name, txt_person_age, txt_person_cc, 
        txt_person_contact, txt_person_email, txt_person_company,
        
        state_enc_front, state_img_front,
        state_enc_left, state_img_left,
        state_enc_right, state_img_right,
        state_enc_glasses, state_img_glasses
    ]
    
    outputs_add_person = [
      
        people_dataframe, remove_people_dataframe, edit_people_dataframe,
       
        txt_person_name, txt_person_age, txt_person_cc, 
        txt_person_contact, txt_person_email, txt_person_company,
 
        state_enc_front, state_img_front,
        state_enc_left, state_img_left,
        state_enc_right, state_img_right,
        state_enc_glasses, state_img_glasses,
        
        snap_front, snap_left, snap_right, snap_glasses
    ]

    btn_people_add_submit.click(
        fn=add_person_and_refresh,
        inputs=inputs_add_person,
        outputs=outputs_add_person
    )

    btn_back_to_people_menu.click(lambda: show_page("people"), None, all_outputs)

    # Remove Person
    inputs_remove_person  = [txt_remove_id]
    outputs_remove_person = [people_dataframe, remove_people_dataframe, edit_people_dataframe, txt_remove_id]
    btn_remove_submit.click(handle_remove_person, inputs_remove_person, outputs_remove_person)
    btn_back_to_people_menu_remove.click(lambda: show_page("people"), None, all_outputs)
    
    outputs_load_person_with_photos = {
        txt_edit_id_display, txt_edit_name, txt_edit_age,
        txt_edit_cc, txt_edit_contact, txt_edit_email, txt_edit_company,
        edit_snap_front, edit_snap_left, edit_snap_right, edit_snap_glasses
    }

    btn_edit_load.click(
        fn=handle_load_person_with_photos, 
        inputs=[txt_edit_id], 
        outputs=outputs_load_person_with_photos
    )  
    
    btn_edit_cap_front.click(
    fn=handle_capture_for_add,
    inputs=[gr.State("front"), edit_state_img_front],
    outputs=[edit_state_enc_front, edit_state_img_front, edit_snap_front]
    )

    btn_edit_cap_left.click(
        fn=handle_capture_for_add,
        inputs=[gr.State("left"), edit_state_img_front],
        outputs=[edit_state_enc_left, edit_state_img_left, edit_snap_left]
    )

    btn_edit_cap_right.click(
        fn=handle_capture_for_add,
        inputs=[gr.State("right"), edit_state_img_front],
        outputs=[edit_state_enc_right, edit_state_img_right, edit_snap_right]
    )

    btn_edit_cap_glasses.click(
        fn=handle_capture_for_add,
        inputs=[gr.State("glasses"), edit_state_img_front],
        outputs=[edit_state_enc_glasses, edit_state_img_glasses, edit_snap_glasses]
    )

    inputs_update_person_with_photos = [
        txt_edit_id_display, 
        txt_edit_age, 
        txt_edit_cc, 
        txt_edit_contact, 
        txt_edit_email, 
        txt_edit_company,
        edit_state_enc_front, 
        edit_state_img_front,
        edit_state_enc_left, 
        edit_state_img_left,
        edit_state_enc_right, 
        edit_state_img_right,
        edit_state_enc_glasses, 
        edit_state_img_glasses
    ]

    outputs_update_person_with_photos = [
        people_dataframe, 
        remove_people_dataframe, 
        edit_people_dataframe
    ]

    btn_edit_submit.click(
        fn=handle_update_person_with_photos,  
        inputs=inputs_update_person_with_photos, 
        outputs=outputs_update_person_with_photos
    )

    btn_back_to_people_menu_edit.click(lambda: show_page("people"), None, all_outputs)

    ## Door Sub-Menu & Forms 
    btn_door_add.click(lambda: show_page("add_door"), None, all_outputs)
    btn_door_remove.click(lambda: show_page("remove_door"), None, all_outputs)
    btn_door_edit.click(lambda: show_page("edit_door"), None, all_outputs)
    btn_back_to_menu_door.click(lambda: show_page("menu"), None, all_outputs)

    # Add Door
    inputs_add_door = [txt_door_location, txt_door_description]
    outputs_add_door = [door_dataframe,remove_door_dataframe,edit_door_dataframe, txt_door_location, txt_door_description]
    btn_door_add_submit.click(
        fn=add_door_and_refresh,
        inputs=inputs_add_door,
        outputs=outputs_add_door
    )
    btn_back_to_door_menu.click(lambda: show_page("door"), None, all_outputs)

    # Remove Door
    inputs_remove_door = [txt_door_remove_id]
    outputs_remove_door = [
        door_dataframe, remove_door_dataframe, edit_door_dataframe,
        txt_door_remove_id
    ]
    btn_door_remove_submit.click(
        fn=handle_remove_door,
        inputs=inputs_remove_door,
        outputs=outputs_remove_door
    )
    btn_back_to_door_menu_remove.click(lambda: show_page("door"), None, all_outputs)

    # Edit Door
    outputs_load_door = {
        txt_edit_door_id_display,
        txt_edit_door_location,
        txt_edit_door_description
    }
    btn_edit_door_load.click(
        fn=handle_load_door,
        inputs=[txt_edit_door_id],
        outputs=outputs_load_door
    )
    inputs_update_door = [txt_edit_door_id_display,txt_edit_door_location, txt_edit_door_description]
    outputs_update_door = [
        door_dataframe, remove_door_dataframe, edit_door_dataframe
    ]
    btn_edit_door_submit.click(
        fn=handle_update_door,
        inputs=inputs_update_door,
        outputs=outputs_update_door
    )
    btn_back_to_door_menu_edit.click(lambda: show_page("door"), None, all_outputs)

    ## Permission Sub-Menu & Forms 
    btn_perm_add.click(lambda: show_page("add_permission"), None, all_outputs)
    btn_perm_remove.click(lambda: show_page("remove_permission"), None, all_outputs)
    btn_perm_edit.click(lambda: show_page("edit_permission"), None, all_outputs)
    btn_back_to_menu_perm.click(lambda: show_page("menu"), None, all_outputs)

    # Add Permission
    inputs_add_perm = [person_dropdown, door_choices]
    outputs_add_perm = [
        permissions_dataframe,
        remove_permissions_dataframe,
        edit_permissions_dataframe,
        person_dropdown,
        door_choices
    ]
    btn_perm_add_submit.click(
        fn=handle_add_permissions,
        inputs=inputs_add_perm,
        outputs=outputs_add_perm
    )
    btn_back_to_perm_menu.click(lambda: show_page("permissions"), None, all_outputs)

    # Remove Permission
    inputs_remove_perm = [txt_perm_remove_id]
    outputs_remove_perm = [
        permissions_dataframe,
        remove_permissions_dataframe,
        edit_permissions_dataframe,
        txt_perm_remove_id
    ]
    btn_perm_remove_submit.click(
        fn=handle_remove_permission,
        inputs=inputs_remove_perm,
        outputs=outputs_remove_perm
    )
    btn_back_to_perm_menu_remove.click(lambda: show_page("permissions"), None, all_outputs)

    # Edit Permission
    btn_perm_edit_load.click(
        fn=handle_load_person_permissions,
        inputs=[edit_person_dropdown],
        outputs=[edit_door_checkboxes]
    )
    inputs_edit_perm = [edit_person_dropdown, edit_door_checkboxes]
    outputs_edit_perm = [
        permissions_dataframe,
        remove_permissions_dataframe,
        edit_permissions_dataframe,
        edit_person_dropdown,
        edit_door_checkboxes
    ]
    btn_perm_edit_submit.click(
        fn=handle_update_person_permissions,
        inputs=inputs_edit_perm,
        outputs=outputs_edit_perm
    )
    btn_back_to_perm_menu_edit.click(lambda: show_page("permissions"), None, all_outputs)

    ## Logs Sub-Menu 
    btn_logs_see.click(lambda: on_click("See Logs"), None, None)
    btn_back_to_menu_logs.click(lambda: show_page("menu"), None, all_outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_error=True)