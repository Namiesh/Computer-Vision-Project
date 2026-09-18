"""
main.py
=======
Main controller script for the gesture drawing computer vision application.
Orchestrates webcam capture, hand tracking, UI interaction, drawing, and display.

Usage:
    python main.py
"""

import time
import cv2
from hand_tracker import HandTracker
from canvas_ui import CanvasUI


def draw_hud(
    frame: cv2.Mat,
    mode_text: str,
    active_color_name: str,
    fps: float,
) -> None:
    """
    Renders status indicators, mode badges, and FPS on the bottom HUD.

    :param frame: Output video frame.
    :param mode_text: Current interaction mode (e.g., 'DRAWING', 'SELECTION', 'STANDBY').
    :param active_color_name: Name of current drawing color.
    :param fps: Calculated frames per second.
    """
    height, width, _ = frame.shape

    # Bottom bar overlay
    hud_y = height - 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, hud_y - 10), (width, height), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Mode Badge
    if mode_text == "DRAWING":
        mode_color = (0, 255, 0)
    elif mode_text == "ERASING":
        mode_color = (0, 165, 255)
    elif mode_text == "SELECTION":
        mode_color = (0, 200, 255)
    else:
        mode_color = (180, 180, 180)

    cv2.putText(
        frame,
        f"MODE: {mode_text}",
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    # Active Color Indicator
    cv2.putText(
        frame,
        f"TOOL: {active_color_name}",
        (250, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Controls Info
    cv2.putText(
        frame,
        "[C] Clear | [E] Eraser | [Q] Quit",
        (width - 440, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # FPS Counter
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (width - 110, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    """Entry point for the gesture drawing application."""
    # 1. Initialize Camera Feed
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    # Request HD resolution (1280x720); falls back gracefully if unsupported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {camera_index}. Please verify camera connection.")
        return

    print("=" * 60)
    print("     GESTURE DRAWING - COMPUTER VISION APPLICATION")
    print("=" * 60)
    print(" [GUIDE] Gesture Controls:")
    print("   * Selection Mode : Raise BOTH Index & Middle fingers")
    print("     - Hover over top buttons to change color or clear canvas")
    print("   * Drawing Mode   : Raise ONLY Index finger")
    print("     - Move index finger to draw in the air")
    print("   * Pause / Standby: Lower fingers or close fist")
    print("   * Key Shortcuts  : Press 'C' to clear canvas | 'Q' or 'ESC' to quit")
    print("=" * 60)

    # 2. Initialize Core Subsystems
    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.65,
    )
    canvas_ui = CanvasUI(brush_thickness=8, header_height=95)

    prev_point = None
    prev_time = time.time()

    window_name = "Gesture Drawing"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Read webcam frame
            success, frame = cap.read()
            if not success or frame is None:
                print("[WARNING] Failed to grab frame from camera stream. Retrying...")
                time.sleep(0.05)
                continue

            # Flip horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)

            # Ensure canvas matrix matches current video resolution
            canvas_ui.init_canvas(frame.shape)

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Hand Detection & Tracking
            frame = tracker.find_hands(frame, draw=False)
            lm_list = tracker.get_landmark_positions(frame)

            mode_text = "STANDBY"

            if lm_list:
                # Determine finger states
                fingers = tracker.get_fingers_up(lm_list)
                index_tip, middle_tip = tracker.get_finger_tips(lm_list)

                if index_tip and middle_tip:
                    x1, y1 = index_tip
                    x2, y2 = middle_tip
                    index_up = fingers[1]
                    middle_up = fingers[2]

                    # ----------------------------------------------------
                    # 1. SELECTION MODE: Both Index & Middle fingers are UP
                    # ----------------------------------------------------
                    if index_up and middle_up:
                        mode_text = "SELECTION"
                        prev_point = None  # Reset stroke history so drawing pauses

                        # Draw selection cursor between index and middle finger tips
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
                        cv2.circle(frame, (cx, cy), 6, (0, 200, 255), cv2.FILLED)

                        # Handle UI button hover/selection using index tip position
                        canvas_ui.handle_selection(x1, y1)

                    # ----------------------------------------------------
                    # 2. DRAWING / ERASING MODE: ONLY Index finger is UP
                    # ----------------------------------------------------
                    elif index_up and not middle_up:
                        is_erasing = (canvas_ui.active_color_name == "ERASER")
                        mode_text = "ERASING" if is_erasing else "DRAWING"

                        # Draw cursor at index finger tip
                        if is_erasing:
                            # Eraser ring cursor indicating erase radius
                            eraser_radius = canvas_ui.eraser_thickness // 2
                            cv2.circle(frame, (x1, y1), eraser_radius, (255, 255, 255), 2)
                            cv2.circle(frame, (x1, y1), 4, (0, 165, 255), cv2.FILLED)
                        else:
                            # Brush cursor
                            cv2.circle(
                                frame,
                                (x1, y1),
                                canvas_ui.brush_thickness,
                                canvas_ui.active_color,
                                cv2.FILLED,
                            )
                            cv2.circle(frame, (x1, y1), canvas_ui.brush_thickness + 2, (255, 255, 255), 2)

                        # Render line segment or eraser stroke on canvas
                        canvas_ui.draw_stroke(prev_point, (x1, y1))
                        prev_point = (x1, y1)

                    # ----------------------------------------------------
                    # 3. STANDBY MODE: Other finger configurations
                    # ----------------------------------------------------
                    else:
                        prev_point = None
            else:
                # No hand detected: reset stroke point to prevent unwanted line jumps
                prev_point = None

            # Composite canvas onto video frame
            output_frame = canvas_ui.merge_canvas(frame)

            # Draw top UI toolbar buttons
            output_frame = canvas_ui.draw_header(output_frame)

            # Draw HUD status bar
            draw_hud(output_frame, mode_text, canvas_ui.active_color_name, fps)

            # Display final composite image
            cv2.imshow(window_name, output_frame)

            # Keyboard Events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to exit
                print("\n[INFO] Exiting application...")
                break
            elif key == ord('c') or key == ord('C'):  # 'c' to clear screen
                canvas_ui.clear_canvas()
            elif key == ord('e') or key == ord('E'):  # 'e' to activate eraser
                canvas_ui.active_color_name = "ERASER"
                canvas_ui.active_color = (0, 0, 0)

    except KeyboardInterrupt:
        print("\n[INFO] Application interrupted by user.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Webcam and OpenCV windows closed.")


if __name__ == "__main__":
    main()
