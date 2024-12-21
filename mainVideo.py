import streamlit as st
import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
from PIL import Image
import altair as alt
import pandas as pd

# Define preprocessing for images
def preprocess_image(img):
    """
    Preprocesses the input test image: applies CLAHE in LAB color space, converts to BGR, 
    and normalizes using the same transformations as training.
    Assumes input is in BGR format.
    """
    # Convert BGR to LAB and apply CLAHE
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])  # Apply CLAHE on the L channel
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Convert back to BGR
    img = Image.fromarray(img)

    # Normalize using the same mean and std as training
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),  # Convert to PIL Image for compatibility with transforms
        transforms.ToTensor(),
        transforms.Normalize([0.5355, 0.4289, 0.3795], [0.2933, 0.2659, 0.2618])
    ])
    return preprocess(img).unsqueeze(0)  # Add batch dimension

# Classes for 6 emotional states
class_names = ['Neutral', 'Fear', 'Anger', 'Happy', 'Surprise', 'Sadness']

# Load the trained model
emotion_model = InceptionResnetV1(pretrained='vggface2', device="cpu", classify=True, num_classes=len(class_names))
emotion_model.load_state_dict(torch.load("emotion_weight.pt", map_location=torch.device('cpu')))
emotion_model.eval()  # Set the model to evaluation mode

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit app
st.title("Real-Time Emotion Detection")
st.text("Press 'Start' to begin video streaming. Press 'Stop' to end.")

# Sidebar buttons
start_button = st.sidebar.button("Start")
stop_button = st.sidebar.button("Stop")

# Placeholder for the video stream and bar chart
video_placeholder = st.empty()
chart_placeholder = st.empty()

# Initialize webcam
if start_button:
    cap = cv2.VideoCapture(0)

    # Stream video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        # Detect faces in the grayscale version of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Prepare a container for class probabilities
        probabilities_list = [0] * len(class_names)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI from the original BGR frame
            face_roi = frame[y:y + h, x:x + w]

            # Preprocess the face image
            face_image = preprocess_image(face_roi)

            with torch.no_grad():
                # Predict emotion using the model
                predictions = emotion_model(face_image)
                probabilities = F.softmax(predictions, dim=1)
                probabilities_list = probabilities[0].tolist()  # Convert tensor to list
                ind = torch.argmax(probabilities).item()
                emotion_label = class_names[ind]
                confidence = probabilities[0][ind].item() * 100

            # Display results on the frame
            cv2.putText(frame, f'Emotion: {emotion_label} {confidence:.2f}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display video frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

                # Display bar chart with probabilities
        if len(faces) != 0:
            # Normalize probabilities to percentage
            probabilities_percent = [int(p * 100) for p in probabilities_list]

            # Create a dynamic UI with boxes
            ui_boxes = ""
            for emotion, prob in zip(class_names, probabilities_percent):
                ui_boxes += f"""
                <div style="
                    display: flex; 
                    align-items: center; 
                    margin-bottom: 10px;">
                    <div style="
                        width: 100px; 
                        text-align: right; 
                        margin-right: 10px;">
                        {emotion}
                    </div>
                    <div style="
                        width: {prob * 3}px; 
                        height: 20px; 
                        background-color: red; 
                        border-radius: 5px;">
                    </div>
                    <div style="
                        margin-left: 10px;">
                        {prob}%
                    </div>
                </div>
                """

            # Display the boxes in Streamlit
            # Display the boxes in Streamlit
            chart_placeholder.markdown(f"""

                {ui_boxes}

            """, unsafe_allow_html=True)


        # Stop if the stop button is pressed
        if stop_button:
            break

    # Release resources
    cap.release()
    st.success("Video streaming stopped.")

st.text("Note: Ensure your webcam is connected and accessible.")
