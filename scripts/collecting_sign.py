import cv2
import asyncio
import os
import mediapipe as mp
import numpy as np

is_collecting_images = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

async def collecting_images(cap, train_test_dir, label, n=20):
    global is_collecting_images
    mp_hands = mp.solutions.hands

    images_dir = os.path.join(train_test_dir, "images")
    landmarks_dir = os.path.join(train_test_dir, "landmarks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(landmarks_dir, exist_ok=True)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        
        await asyncio.sleep(1.5) #time.sleep(2)
        for i in range(n):
            _, frame = cap.read()

            # process lanmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks_results = hands.process(frame_rgb)


            landmarks = np.full((2, 21, 3), np.nan, dtype=np.float32)

            # saving the landmarks
            if landmarks_results.multi_hand_landmarks:
                for h_idx, hand_lms in enumerate(landmarks_results.multi_hand_landmarks):
                    if h_idx >= 2:
                        break  # safety, though max_num_hands=2
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=4, circle_radius=10),
                        mp_drawing.DrawingSpec(color=(128, 64, 128), thickness=4)
                    )
                    landmarks[h_idx] = np.array(
                        [[p.x, p.y, p.z] for p in hand_lms.landmark],
                        dtype=np.float32
                    )

            # save image
            image_path = os.path.join(
                images_dir, f"label_{label}_image_{i}.jpg"
            )
            cv2.imwrite(image_path, frame)
        

            landmarks_path = os.path.join(
                landmarks_dir, f"label_{label}_image_{i}.npy"
            )
            np.save(landmarks_path, landmarks)
            # writing a red border around the frame so that we know its capturing
            h, w, _ = frame.shape
            border_frame = frame.copy()
            cv2.rectangle(border_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            #cv2.putText(frame, f"Recording '{label}'  {i+1}/{n}",
                        #(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(border_frame, f"Recording '{label}'  {i+1}/{n}", (w - 520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("MediaPipe Hands", border_frame)
            cv2.waitKey(1)
            
            await asyncio.sleep(1.5)
    is_collecting_images = False

# Webcam
cap = cv2.VideoCapture(0)

# labels to train : 
labels = []
current_label = 0
train_test_dir = './data/'

while cap.isOpened():
        
    success, image = cap.read()
    h, w, _ = image.shape

    # key is the key you press
    key = cv2.waitKey(1) & 0xFF


    # Collecting images
    if key == ord('d'):
        
        cv2.putText(image, f"Collecting images for label {len(labels)}", (w - 520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        is_collecting_images = True
        asyncio.run(collecting_images(cap, train_test_dir, label=len(labels), n=20)) # collecting images without blocking the frames
        labels.append(current_label)
        current_label+=1

    if key == 27:  # key == ESC
        break

    if is_collecting_images == False :
        cv2.putText(image, 'Press D to start collecting a sign', (w - 520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    
    cv2.imshow("MediaPipe Hands", image) # open window
    
cap.release()
