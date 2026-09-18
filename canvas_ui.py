"""
canvas_ui.py
=============
Module responsible for the drawing UI overlay, color selection state,
drawing surface management, and frame blending.
"""

from typing import Tuple, Optional, Dict, List
import cv2
import numpy as np


class CanvasUI:
    """
    Manages the drawing UI elements, color palette, canvas drawing buffer,
    and rendering overlays on the video stream.
    """

    # BGR Color definitions
    COLORS: Dict[str, Tuple[int, int, int]] = {
        "BLUE": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
    }

    def __init__(
        self,
        brush_thickness: int = 8,
        eraser_thickness: int = 50,
        header_height: int = 100,
    ):
        """
        Initializes the CanvasUI.

        :param brush_thickness: Stroke thickness in pixels for drawing.
        :param eraser_thickness: Stroke thickness in pixels for erasing.
        :param header_height: Height of the top toolbar in pixels.
        """
        self.brush_thickness = brush_thickness
        self.eraser_thickness = eraser_thickness
        self.header_height = header_height

        # Default drawing color
        self.active_color_name = "BLUE"
        self.active_color = self.COLORS[self.active_color_name]

        # Canvas matrix (initialized when frame dimensions are known)
        self.canvas: Optional[np.ndarray] = None
        self.buttons: List[Dict] = []
        self._frame_width = 0
        self._frame_height = 0

    def init_canvas(self, frame_shape: Tuple[int, int, int]) -> None:
        """
        Initializes or resizes the canvas matrix and sets up UI button regions.

        :param frame_shape: Tuple representing (height, width, channels).
        """
        height, width, _ = frame_shape
        if self.canvas is None or (self._frame_height != height or self._frame_width != width):
            self._frame_height = height
            self._frame_width = width
            self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
            self._setup_buttons(width)

    def _setup_buttons(self, width: int) -> None:
        """
        Calculates 6 distinct rectangular bounding boxes across the top header.
        Order: CLEAR, BLUE, GREEN, RED, YELLOW, ERASER.

        :param width: Width of the camera frame.
        """
        self.buttons = []
        num_buttons = 6
        margin_x = 12
        margin_y = 10
        btn_height = self.header_height - (margin_y * 2)

        total_margin_space = margin_x * (num_buttons + 1)
        btn_width = (width - total_margin_space) // num_buttons

        button_specs = [
            {"id": "CLEAR", "label": "CLEAR SCREEN", "color": (50, 50, 50), "text_color": (255, 255, 255)},
            {"id": "BLUE", "label": "BLUE", "color": self.COLORS["BLUE"], "text_color": (255, 255, 255)},
            {"id": "GREEN", "label": "GREEN", "color": self.COLORS["GREEN"], "text_color": (0, 0, 0)},
            {"id": "RED", "label": "RED", "color": self.COLORS["RED"], "text_color": (255, 255, 255)},
            {"id": "YELLOW", "label": "YELLOW", "color": self.COLORS["YELLOW"], "text_color": (0, 0, 0)},
            {"id": "ERASER", "label": "ERASER", "color": (80, 80, 90), "text_color": (255, 255, 255)},
        ]

        for i, spec in enumerate(button_specs):
            x1 = margin_x + i * (btn_width + margin_x)
            y1 = margin_y
            x2 = x1 + btn_width
            y2 = y1 + btn_height

            self.buttons.append({
                "id": spec["id"],
                "label": spec["label"],
                "bbox": (x1, y1, x2, y2),
                "bg_color": spec["color"],
                "text_color": spec["text_color"],
            })

    def handle_selection(self, x: int, y: int) -> Optional[str]:
        """
        Checks if the given point is inside any UI button bounding box.
        Updates the active color or clears the canvas accordingly.

        :param x: X-coordinate of finger tip.
        :param y: Y-coordinate of finger tip.
        :return: Action/Button ID if a button was clicked, None otherwise.
        """
        if y > self.header_height:
            return None

        for btn in self.buttons:
            x1, y1, x2, y2 = btn["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                btn_id = btn["id"]
                if btn_id == "CLEAR":
                    self.clear_canvas()
                    return "CLEAR"
                elif btn_id == "ERASER":
                    self.active_color_name = "ERASER"
                    self.active_color = (0, 0, 0)
                    return "ERASER"
                elif btn_id in self.COLORS:
                    self.active_color_name = btn_id
                    self.active_color = self.COLORS[btn_id]
                    return btn_id

        return None

    def draw_stroke(
        self,
        prev_point: Optional[Tuple[int, int]],
        curr_point: Tuple[int, int],
    ) -> None:
        """
        Draws a continuous line segment (or eraser stroke) on the canvas matrix.

        :param prev_point: Starting (x, y) coordinate.
        :param curr_point: Ending (x, y) coordinate.
        """
        if self.canvas is None:
            return

        # Avoid drawing into the top header toolbar area
        if curr_point[1] <= self.header_height:
            return

        is_eraser = (self.active_color_name == "ERASER")
        thickness = self.eraser_thickness if is_eraser else self.brush_thickness
        stroke_color = (0, 0, 0) if is_eraser else self.active_color

        if prev_point is None:
            # Draw an initial dot if starting a stroke
            cv2.circle(
                self.canvas,
                curr_point,
                thickness // 2,
                stroke_color,
                cv2.FILLED,
            )
        else:
            # Do not connect stroke if previous point was inside header
            if prev_point[1] > self.header_height:
                cv2.line(
                    self.canvas,
                    prev_point,
                    curr_point,
                    stroke_color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    self.canvas,
                    curr_point,
                    thickness // 2,
                    stroke_color,
                    cv2.FILLED,
                )

    def clear_canvas(self) -> None:
        """Resets the drawing canvas to empty black."""
        if self.canvas is not None:
            self.canvas[:] = 0

    def draw_header(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Renders the top toolbar with 5 buttons, highlighting the active tool.

        :param frame: Target frame to draw buttons on.
        :return: Frame with rendered UI header.
        """
        # Create subtle translucent dark banner behind header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._frame_width, self.header_height), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw each button
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["bbox"]
            btn_id = btn["id"]
            is_active = (btn_id == self.active_color_name)

            # Draw button background
            cv2.rectangle(frame, (x1, y1), (x2, y2), btn["bg_color"], cv2.FILLED)

            # Draw border (highlight active button with white double-thickness border)
            border_color = (255, 255, 255) if is_active else (80, 80, 80)
            border_thickness = 4 if is_active else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)

            # Draw button text label centered
            label = btn["label"]
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            font_thickness = 1 if not is_active else 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2

            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                font,
                font_scale,
                btn["text_color"],
                font_thickness,
                cv2.LINE_AA,
            )

        return frame

    def merge_canvas(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Merges the drawing canvas with the live webcam frame.

        :param frame: Live webcam frame.
        :return: Merged composite frame.
        """
        if self.canvas is None:
            return frame

        # Convert canvas to grayscale and threshold to isolate drawn strokes
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, stroke_mask_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)

        # Clear background frame where canvas has strokes
        frame_bg = cv2.bitwise_and(frame, frame, mask=stroke_mask_inv)

        # Overlay canvas strokes onto the background frame
        merged_frame = cv2.bitwise_or(frame_bg, self.canvas)
        return merged_frame
