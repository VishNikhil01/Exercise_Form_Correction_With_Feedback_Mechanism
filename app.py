import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import datetime
import time
import pygame
import torch
import pickle
import signal
import sys

st.set_page_config(
    page_title="Real-time Big Three Workout AI Posture Correction Service",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Load YOLOv5 model
model_weights_path = "D:/ISL-MAJOR-PROJECT/models/best_big_bounding.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Record previous alert time
previous_alert_time = 0
counter = 0

def most_frequent(data):
    return max(data, key=data.count)


# Angle calculation function
def calculateAngle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Object detection function using YOLOv5
def detect_objects(frame):
    results = model(frame)
    pred = results.pred[0]

    return pred


# Streamlit app initialization
st.title("Real-time Big Three Workout AI Posture Correction Service")

pygame.mixer.init()

# Add menu to sidebar
menu_selection = st.selectbox("Select Exercise", ("Bench Press", "Squat", "Deadlift"))
counter_display = st.sidebar.empty()
counter_display.header(f"Current Counter: {counter} reps")

# Load different models based on the selected exercise

current_stage = ""
posture_status = [None]

model_weights_path = "D:/ISL-MAJOR-PROJECT/models/benchpress/benchpress.pkl"
with open(model_weights_path, "rb") as f:
    model_e = pickle.load(f)

if menu_selection == "Bench Press":
    model_weights_path = "D:/ISL-MAJOR-PROJECT/models/benchpress/benchpress.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)
elif menu_selection == "Squat":
    model_weights_path = "D:/ISL-MAJOR-PROJECT/models/Sqat/squat.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)
elif menu_selection == "Deadlift":
    model_weights_path = "D:/ISL-MAJOR-PROJECT/models/deadlift/deadlift.pkl"
    with open(model_weights_path, "rb") as f:
        model_e = pickle.load(f)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Reduce video resolution to 640x480 for better performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Skip every 5th frame to reduce processing load
frame_skip_interval = 5
frame_counter = 0

# Initialize Mediapipe Pose model: min detection confidence=0.5, min tracking confidence=0.5, model complexity=1
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Joint Detection Confidence Threshold", 0.0, 1.0, 0.7)

# Initialize empty areas for angle display
neck_angle_display = st.sidebar.empty()
left_shoulder_angle_display = st.sidebar.empty()
right_shoulder_angle_display = st.sidebar.empty()
left_elbow_angle_display = st.sidebar.empty()
right_elbow_angle_display = st.sidebar.empty()
left_hip_angle_display = st.sidebar.empty()
right_hip_angle_display = st.sidebar.empty()
left_knee_angle_display = st.sidebar.empty()
right_knee_angle_display = st.sidebar.empty()
left_ankle_angle_display = st.sidebar.empty()
right_ankle_angle_display = st.sidebar.empty()

# Remove signal.signal and use a flag for graceful termination
running = True

def stop_running():
    global running
    running = False

# Add a Streamlit button to stop the application
if st.sidebar.button("Stop Application"):
    stop_running()

# Check if the webcam is successfully opened
if not camera.isOpened():
    st.error("Unable to access the webcam. Please ensure it is connected and not in use by another application.")
    running = False

while running:
    try:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame from webcam. Please check your webcam connection.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)  # Flip frame horizontally

        # Object detection using YOLOv5
        results_yolo = detect_objects(frame)

        # Display YOLOv5 results on screen
        try:
            if results_yolo is not None:
                for det in results_yolo:
                    c1, c2 = det[:2].int(), det[2:4].int()
                    cls, conf, *_ = det
                    label = f"person {conf:.2f}"

                    if conf >= 0.7:  # Display object only if confidence is 0.7 or higher
                        # Convert c1 and c2 to tuples
                        c1 = (c1[0].item(), c1[1].item())
                        c2 = (c2[0].item(), c2[1].item())

                        # Extract the frame of the detected object
                        object_frame = frame[c1[1] : c2[1], c1[0] : c2[0]]

                        # Process the object frame for Pose estimation
                        object_frame_rgb = cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB)
                        results_pose = pose.process(object_frame_rgb)

                        if results_pose.pose_landmarks is not None:
                            landmarks = results_pose.pose_landmarks.landmark
                            nose = [
                                landmarks[mp_pose.PoseLandmark.NOSE].x,
                                landmarks[mp_pose.PoseLandmark.NOSE].y,
                            ]  # Nose
                            left_shoulder = [
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                            ]  # Left Shoulder
                            left_elbow = [
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                            ]  # Left Elbow
                            left_wrist = [
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
                            ]  # Left Wrist
                            left_hip = [
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                            ]  # Left Hip
                            left_knee = [
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                            ]  # Left Knee
                            left_ankle = [
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                            ]  # Left Ankle
                            left_heel = [
                                landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,
                            ]  # Left Heel
                            right_shoulder = [
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                            ]  # Right Shoulder
                            right_elbow = [
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                            ]  # Right Elbow
                            right_wrist = [
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                            ]  # Right Wrist
                            right_hip = [
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                            ]  # Right Hip
                            right_knee = [
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                            ]  # Right Knee
                            right_ankle = [
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                            ]  # Right Ankle
                            right_heel = [
                                landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y,
                            ]  # Right Heel

                            # Calculate angles
                            neck_angle = (
                                calculateAngle(left_shoulder, nose, left_hip)
                                + calculateAngle(right_shoulder, nose, right_hip) / 2
                            )
                            left_elbow_angle = calculateAngle(
                                left_shoulder, left_elbow, left_wrist
                            )
                            right_elbow_angle = calculateAngle(
                                right_shoulder, right_elbow, right_wrist
                            )
                            left_shoulder_angle = calculateAngle(
                                left_elbow, left_shoulder, left_hip
                            )
                            right_shoulder_angle = calculateAngle(
                                right_elbow, right_shoulder, right_hip
                            )
                            left_hip_angle = calculateAngle(
                                left_shoulder, left_hip, left_knee
                            )
                            right_hip_angle = calculateAngle(
                                right_shoulder, right_hip, right_knee
                            )
                            left_knee_angle = calculateAngle(
                                left_hip, left_knee, left_ankle
                            )
                            right_knee_angle = calculateAngle(
                                right_hip, right_knee, right_ankle
                            )
                            left_ankle_angle = calculateAngle(
                                left_knee, left_ankle, left_heel
                            )
                            right_ankle_angle = calculateAngle(
                                right_knee, right_ankle, right_heel
                            )

                            # Update angle displays
                            neck_angle_display.text(f"Neck Angle: {neck_angle:.2f}°")
                            left_shoulder_angle_display.text(
                                f"Left Shoulder Angle: {left_shoulder_angle:.2f}°"
                            )
                            right_shoulder_angle_display.text(
                                f"Right Shoulder Angle: {right_shoulder_angle:.2f}°"
                            )
                            left_elbow_angle_display.text(
                                f"Left Elbow Angle: {left_elbow_angle:.2f}°"
                            )
                            right_elbow_angle_display.text(
                                f"Right Elbow Angle: {right_elbow_angle:.2f}°"
                            )
                            left_hip_angle_display.text(
                                f"Left Hip Angle: {left_hip_angle:.2f}°"
                            )
                            right_hip_angle_display.text(
                                f"Right Hip Angle: {right_hip_angle:.2f}°"
                            )
                            left_knee_angle_display.text(
                                f"Left Knee Angle: {left_knee_angle:.2f}°"
                            )
                            right_knee_angle_display.text(
                                f"Right Knee Angle: {right_knee_angle:.2f}°"
                            )
                            left_ankle_angle_display.text(
                                f"Left Ankle Angle: {left_ankle_angle:.2f}°"
                            )
                            right_ankle_angle_display.text(
                                f"Right Ankle Angle: {right_ankle_angle:.2f}°"
                            )

                    # Rep counting algorithm implementation
                    # Add debug statements to verify posture_status and feedback conditions
                    try:
                        row = [
                            coord
                            for res in results_pose.pose_landmarks.landmark
                            for coord in [res.x, res.y, res.z, res.visibility]
                        ]
                        X = pd.DataFrame([row])
                        exercise_class = model_e.predict(X)[0]
                        exercise_class_prob = model_e.predict_proba(X)[0]
                        print(f"Exercise class: {exercise_class}, Probabilities: {exercise_class_prob}")

                        if "down" in exercise_class:
                            current_stage = "down"
                            posture_status.append(exercise_class)
                            print(f"Posture status updated: {posture_status}")
                        elif current_stage == "down" and "up" in exercise_class:
                            current_stage = "up"
                            counter += 1
                            posture_status.append(exercise_class)
                            print(f"Posture status updated: {posture_status}")
                            counter_display.header(f"Current count: {counter} times")

                            if "correct" not in most_frequent(posture_status):
                                current_time = time.time()
                                if current_time - previous_alert_time >= 3:
                                    now = datetime.datetime.now()
                                    print(f"Most frequent posture: {most_frequent(posture_status)}")
                                    if "excessive_arch" in most_frequent(posture_status):
                                        st.error("Avoid arching your lower back too much; try to keep it natural.")
                                        pygame.mixer.music.load("D:/ISL-MAJOR-PROJECT/resources/sounds/excessive_arch_1.mp3")
                                        pygame.mixer.music.play()
                                        posture_status = []
                                        previous_alert_time = current_time
                                    elif "arms_spread" in most_frequent(posture_status):
                                        st.error("Your grip is too wide. Hold the bar a bit narrower.")
                                        pygame.mixer.music.load("D:/ISL-MAJOR-PROJECT/resources/sounds/arms_spread_1.mp3")
                                        pygame.mixer.music.play()
                                        posture_status = []
                                        previous_alert_time = current_time
                                    elif "spine_neutral" in most_frequent(posture_status):
                                        st.error("Avoid excessive curvature of the spine.")
                                        pygame.mixer.music.load("D:/ISL-MAJOR-PROJECT/resources/sounds/spine_neutral_feedback_1.mp3")
                                        pygame.mixer.music.play()
                                        posture_status = []
                                        previous_alert_time = current_time
                                    elif "caved_in_knees" in most_frequent(posture_status):
                                        st.error("Be cautious not to let your knees cave in during the squat.")
                                        pygame.mixer.music.load("D:/ISL-MAJOR-PROJECT/resources/sounds/caved_in_knees_feedback_1.mp3")
                                        pygame.mixer.music.play()
                                        posture_status = []
                                        previous_alert_time = current_time
                                    elif "feet_spread" in most_frequent(posture_status):
                                        st.error("Narrow your stance to about shoulder width.")
                                        pygame.mixer.music.load("D:/ISL-MAJOR-PROJECT/resources/sounds/feet_spread.mp3")
                                        pygame.mixer.music.play()
                                        posture_status = []
                                        previous_alert_time = current_time
                                    elif "arms_narrow" in most_frequent(posture_status):
                                        st.error("Your grip is too wide. Hold the bar a bit narrower.")
                                        pygame.mixer.music.load("D:/ISL-MAJOR-PROJECT/resources/sounds/arms_narrow.mp3")
                                        pygame.mixer.music.play()
                                        posture_status = []
                                        previous_alert_time = current_time
                        elif "correct" in most_frequent(posture_status):
                            pygame.mixer.music.load("D:/ISL-MAJOR-PROJECT/resources/sounds/correct.mp3")
                            pygame.mixer.music.play()
                            st.info("You are performing the exercise with the correct posture.")
                            posture_status = []
                    except Exception as e:
                        print(f"Error in feedback logic: {e}")
                        pass

                    # Draw landmarks
                    for landmark in mp_pose.PoseLandmark:
                        if landmarks[landmark.value].visibility >= confidence_threshold:
                            mp.solutions.drawing_utils.draw_landmarks(
                                object_frame,
                                results_pose.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                            )

                # Draw the object frame back onto the original frame
                frame = object_frame

            # Display the original frame
            FRAME_WINDOW.image(frame)
        except Exception as e:
            pass
    except KeyboardInterrupt:
        stop_running()