"""
hand_tracker.py
================
Module responsible for hand detection and landmark tracking using MediaPipe.
Encapsulates MediaPipe Hands setup, frame inference, landmark extraction,
and finger state estimation (raised / lowered).
"""

from typing import List, Tuple, Optional
import cv2
import mediapipe as mp


class HandTracker:
    """
    Tracks hand landmarks using MediaPipe and evaluates finger states.
    """


    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        """
        Initializes the MediaPipe Hands solution.

        :param static_image_mode: Treat frames as independent images if True.
        :param max_num_hands: Maximum number of hands to detect.
        :param min_detection_confidence: Confidence threshold for hand detection.
        :param min_tracking_confidence: Confidence threshold for landmark tracking.
        """
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        except AttributeError:
            raise ImportError(
                "MediaPipe 'solutions' module was not found. "
                "Please ensure you are using a compatible version by running: "
                "pip install mediapipe==0.10.14"
            )

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.results = None

    def find_hands(self, frame: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """
        Processes a BGR frame to detect hands and optionally draws landmarks.

        :param frame: BGR input image from webcam.
        :param draw: Whether to render landmark connectors on the frame.
        :return: Frame with landmarks drawn (if draw=True).
        """
        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self.results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        return frame

    def get_landmark_positions(
        self, frame: cv2.Mat, hand_index: int = 0
    ) -> List[Tuple[int, int, int]]:
        """
        Extracts pixel coordinates for all 21 landmarks of a detected hand.

        :param frame: The frame used to scale normalized coordinates to pixels.
        :param hand_index: Index of the hand (default is 0 for primary hand).
        :return: List of tuples [(landmark_id, x, y), ...] or empty list if no hand detected.
        """
        landmark_list: List[Tuple[int, int, int]] = []

        if self.results and self.results.multi_hand_landmarks:
            if hand_index < len(self.results.multi_hand_landmarks):
                selected_hand = self.results.multi_hand_landmarks[hand_index]
                height, width, _ = frame.shape

                for lm_id, landmark in enumerate(selected_hand.landmark):
                    # Convert normalized coordinates [0.0, 1.0] to pixel values
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append((lm_id, cx, cy))

        return landmark_list

    def get_fingers_up(self, lm_list: List[Tuple[int, int, int]]) -> List[bool]:
        """
        Determines which fingers are currently raised.

        :param lm_list: Landmark list [(id, x, y), ...] containing at least 21 points.
        :return: List of booleans [Thumb, Index, Middle, Ring, Pinky] (True = Up).
        """
        if len(lm_list) < 21:
            return [False, False, False, False, False]

        fingers = []

        # Thumb: Compare x-coordinate of tip with the joint for horizontal thumb check
        # Handles mirrored coordinate orientation
        if lm_list[self.TIP_IDS[0]][1] > lm_list[self.TIP_IDS[0] - 1][1]:
            fingers.append(True)
        else:
            fingers.append(False)

        # 4 Fingers (Index, Middle, Ring, Pinky):
        # In image space, y=0 is top. A finger is raised if tip.y < pip.y
        for i in range(1, 5):
            tip_id = self.TIP_IDS[i]
            pip_id = self.PIP_IDS[i]
            if lm_list[tip_id][2] < lm_list[pip_id][2]:
                fingers.append(True)
            else:
                fingers.append(False)

        return fingers

    def get_finger_tips(
        self, lm_list: List[Tuple[int, int, int]]
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Returns pixel coordinates for Index Finger Tip (Landmark 8) and
        Middle Finger Tip (Landmark 12).

        :param lm_list: Landmark list.
        :return: Tuple ((x8, y8), (x12, y12)) or (None, None) if not available.
        """
        if len(lm_list) < 21:
            return None, None

        index_tip = (lm_list[8][1], lm_list[8][2])
        middle_tip = (lm_list[12][1], lm_list[12][2])
        return index_tip, middle_tip
