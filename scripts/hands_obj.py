import math
import numpy as np


class HandRot:
    def __init__(self):
        self.hand_closed_old = None
        self.angle_rotation_old = None
        self.angle_front_orientation_old = None
        self.angle_wrist_orientation_old = None

        self.hand_closed_new = None
        self.angle_rotation_new = None
        self.angle_front_orientation_new = None
        self.angle_wrist_orientation_new = None

        self.hand_landmarks = None

    def set_hand_rotation_deg(self):
        wrist = self.hand_landmarks.landmark[0]
        middle_mcp = self.hand_landmarks.landmark[9]

        dx = middle_mcp.x - wrist.x
        dy = wrist.y - middle_mcp.y
        angle = math.degrees(math.atan2(dy, dx))

        self.angle_rotation_old = self.angle_rotation_new
        self.angle_rotation_new = (angle + 360) % 360

    def set_hand_front_orientation_deg(self):
        def _v(lm):
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        wrist = _v(self.hand_landmarks.landmark[0])
        index_mcp = _v(self.hand_landmarks.landmark[5])
        pinky_mcp = _v(self.hand_landmarks.landmark[17])

        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        n = np.cross(v1, v2)
        n = n / (np.linalg.norm(n) + 1e-9)

        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cosang = float(np.clip(np.dot(n, z_axis), -1.0, 1.0))
        angle = math.degrees(math.acos(cosang))

        self.angle_front_orientation_old = self.angle_front_orientation_new
        self.angle_front_orientation_new = angle

    def set_hand_wrist_orientation_deg(self):
        index_mcp = self.hand_landmarks.landmark[5]
        pinky_mcp = self.hand_landmarks.landmark[17]

        dx = pinky_mcp.x - index_mcp.x
        dy = index_mcp.y - pinky_mcp.y
        angle = math.degrees(math.atan2(dy, dx))

        self.angle_wrist_orientation_old = self.angle_wrist_orientation_new
        self.angle_wrist_orientation_new = (angle + 360) % 360

    def set_is_hand_closed(self, axis="y"):
        tip = self.hand_landmarks.landmark[12]
        mcp = self.hand_landmarks.landmark[9]
        wrist = self.hand_landmarks.landmark[0]

        if axis == "y":
            closed = abs(tip.y - wrist.y) < abs(mcp.y - wrist.y)
        else:
            closed = abs(tip.x - wrist.x) < abs(mcp.x - wrist.x)

        self.hand_closed_old = self.hand_closed_new
        self.hand_closed_new = closed

class EyesRot:
    # This class is not used by the main python file, however you can use it to move your cursor with the eyes.
    def __init__(self):
        self.hand_closed_old = None
        self.angle_rotation_old = None
        self.angle_front_orientation_old = None
        self.angle_wrist_orientation_old = None

        self.hand_closed_new = None
        self.angle_rotation_new = None
        self.angle_front_orientation_new = None
        self.angle_wrist_orientation_new = None

        self.hand_landmarks = None