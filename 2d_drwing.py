import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Variables to store previous drawing position, drawing state, and selected color
prev_x, prev_y = None, None
drawing = False
current_color = (0, 0, 0)  # Default color is black
brush_thickness = 5
eraser_thickness = 20  # Increased thickness for eraser
canvas = None  # Initialize the canvas as None

# Colors list and color palette positions
colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
color_palette_positions = [(50 + i * 100, 50) for i in range(len(colors))]
color_index = 0

# Variables for gesture detection
last_toggle_time = 0
gesture_timeout = 1  # Time in seconds to debounce the gesture detection
color_change_timeout = 2  # Time in seconds to debounce color change

# Create a named window and set it to fullscreen
cv2.namedWindow("2D Paint", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("2D Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Create canvas if it doesn't exist yet or if its size doesn't match the frame
    if canvas is None or canvas.shape[1] != frame.shape[1] or canvas.shape[0] != frame.shape[0]:
        canvas = np.ones((frame.shape[0], frame.shape[1], 3), dtype="uint8") * 255

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(rgb_frame)

    # Draw the color palette on the canvas
    for i, (x_pos, y_pos) in enumerate(color_palette_positions):
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 80, y_pos + 40), colors[i], -1)
        cv2.putText(frame, f"{i + 1}", (x_pos + 35, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # If hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger (landmark 8)
            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            # Get the tip of the thumb (landmark 4)
            thumb_x = int(hand_landmarks.landmark[4].x * frame.shape[1])
            thumb_y = int(hand_landmarks.landmark[4].y * frame.shape[0])

            # Calculate distance between thumb and index finger
            distance = np.hypot(x - thumb_x, y - thumb_y)

            # Check if the index finger and thumb are close enough to consider it a clicking gesture
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if distance < 40 and (current_time - last_toggle_time) > gesture_timeout:
                drawing = not drawing  # Toggle drawing state
                last_toggle_time = current_time  # Update the last toggle time
                prev_x, prev_y = None, None  # Reset previous position

            # If in drawing mode and previous coordinates exist, draw a line between previous and current points
            if drawing and prev_x is not None and prev_y is not None:
                thickness = eraser_thickness if current_color == (255, 255, 255) else brush_thickness
                cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness)

            # Update the previous coordinates
            prev_x, prev_y = x, y

            # Check if the index finger is near the color palette to change color
            for i, (x_pos, y_pos) in enumerate(color_palette_positions):
                if x_pos < x < x_pos + 80 and y_pos < y < y_pos + 40:
                    if (current_time - last_toggle_time) > color_change_timeout:
                        current_color = colors[i]
                        last_toggle_time = current_time  # Update the last toggle time
                        prev_x, prev_y = None, None  # Reset previous position

            # Check if the index finger is near the bottom of the screen to activate eraser
            if y > frame.shape[0] - 50:
                current_color = (255, 255, 255)  # Set color to white for erasing
                prev_x, prev_y = None, None  # Reset previous position

    # If no hand is detected, reset previous coordinates
    else:
        prev_x, prev_y = None, None

    # Combine the frame with the canvas
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Add text prompts to the frame
    if drawing:
        cv2.putText(combined_frame, "Drawing Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(combined_frame, "Not Drawing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.putText(combined_frame, f"Color: {current_color}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2, cv2.LINE_AA)
    
    if current_color == (255, 255, 255):
        cv2.putText(combined_frame, "Eraser Activated", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("2D Paint", combined_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
