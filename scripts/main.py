import cv2
import mediapipe as mp
import numpy as np
import math
from hands_obj import HandRot
from pynput.mouse import Controller
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

mouse = Controller()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# MLP Classifier
with open('model/mlp_classifier.pkl', 'rb') as f:
    mlp = pickle.load(f)
with open('./model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


LABELS_MAPPING = {
    "heart_1": 0,
    "heart_2": 1,
    "heart_3": 2,
    "index_lifted": 3,
    "f*ck_sign": 4,
    "victory_sign": 5,
    "ok": 6,
    "fist": 7,
    "middle_finger_touching_thumb": 8,
    "little_finger_up": 9,
    "ring_finger_touching_thumb": 10,
    "triangle": 11,
    "thumbs_up": 12,
    'None': None
}

def one_hot_to_label(vec) -> int:
    vec = np.asarray(vec)

    if vec.ndim == 2:
        vec = vec[0]

    if vec.sum() == 0:
        return None

    return int(vec.argmax())

LABELS_MAPPING_REV = {v:k for k, v in LABELS_MAPPING.items()}




def hand_bbox_px(hand_landmarks, w, h) -> tuple[int, int, int, int]:
    """Calculate the bounding box coordinates of a hand (pixel values)"""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    x1 = int(max(0, min(xs) * w))
    y1 = int(max(0, min(ys) * h))
    x2 = int(min(w - 1, max(xs) * w))
    y2 = int(min(h - 1, max(ys) * h))
    return x1, y1, x2, y2

def measure_text_box(lines, font, scale, thickness, line_gap, margin):
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    tw = max(s[0] for s in sizes) if sizes else 0
    th = sum(s[1] for s in sizes) + (len(lines) - 1) * line_gap if sizes else 0
    box_w = tw + 2 * margin
    box_h = th + 2 * margin
    return sizes, box_w, box_h, tw, th

def draw_box(frame, text, side="right", y=None, panel_margin=20, color=(255, 0, 0)):
    """
    Draw a box on the video frame with text. It handles '\n' character. 
    Params : 
      - side: 'left' or 'right'
      - y: baseline position (bottom of box). If None -> bottom margin.
    Returns : the box height so you can stack multiple boxes.
    """
    h, w, _ = frame.shape
    if y is None:
        y = h - 20

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    line_gap = 28
    margin = 10

    lines = text.split("\n")

    sizes, box_w, box_h, tw, th = measure_text_box(lines, font, scale, thickness, line_gap, margin)

    # Choose x based on side
    if side == "left":
        x = panel_margin
    else:
        x = w - panel_margin - box_w

    # Background rectangle
    top_left = (x, y - box_h)
    bottom_right = (x + box_w, y)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)

    # Draw lines
    yy = y - box_h + margin + (sizes[0][1] if sizes else 0)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + margin, yy), font, scale, color, thickness, cv2.LINE_AA)
        yy += line_gap

    return box_h


# Webcam
cap = cv2.VideoCapture(0)

# Initialisation
left_hand = HandRot()
right_hand = HandRot()


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)

        h, w, _ = image.shape

        # Process
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True


        if results.multi_hand_landmarks:
            left_y = h - 20
            right_y = h - 20

            for hand_landmarks in results.multi_hand_landmarks:
                

                # Get the side of the screen where the hand is in by bbox center
                x1, y1, x2, y2 = hand_bbox_px(hand_landmarks, w, h)
                cx = (x1 + x2) / 2.0
                side = "left" if cx < (w / 2) else "right"
                
                if side=="left":

                    #draws landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(128, 128, 128),
                            thickness=4,
                            circle_radius=10,
                        ),
                        mp_drawing.DrawingSpec(
                            color=(128, 64, 128),
                            thickness=4,
                        ),
                    )

                    # formatting input
                    flat = [v for lm in hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
                    flat = scaler.transform([flat])
                    y_pred = mlp.predict([flat[0]])

                    text = (f"Hand sign detected : {LABELS_MAPPING_REV[one_hot_to_label(y_pred)]}\n")

                    box_h = draw_box(image, text, side="left", y=left_y, color=(121, 0, 121))
                    left_y -= (box_h + 10)
                
                else: # right

                    # draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(128, 128, 128),
                            thickness=4,
                            circle_radius=10,
                        ),
                        mp_drawing.DrawingSpec(
                            color=(200, 70, 40),
                            thickness=4,
                        ),
                    )
                    
                    right_hand.hand_landmarks = hand_landmarks
                    right_hand.set_hand_front_orientation_deg()
                    right_hand.set_hand_rotation_deg()
                    right_hand.set_hand_wrist_orientation_deg()
                    right_hand.set_is_hand_closed()

                    text = (
                        f"Rotation angle : {right_hand.angle_rotation_new:.1f}°\n"
                        f"Wrist angle : {right_hand.angle_wrist_orientation_new:.1f}°\n"
                        f"Front orientation angle : {right_hand.angle_front_orientation_new:.1f}°\n"
                        f"Hand closed : {right_hand.hand_closed_new}\nRight"
                    )

                    box_h = draw_box(image, text, side="right", y=right_y)
                    right_y -= (box_h + 10)
                
                    # mouse moving with right hand :
                    if right_hand.hand_closed_new == False and right_hand.angle_wrist_orientation_old != None:
                        move_side =  right_hand.angle_rotation_old - right_hand.angle_rotation_new
                        move_down = right_hand.angle_front_orientation_new - right_hand.angle_front_orientation_old
                        x, y = mouse.position
                        mouse.position = (x + move_side *100, y + (move_down * 50))

        cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
