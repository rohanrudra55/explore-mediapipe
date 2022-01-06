import cv2
import mediapipe as mp

class Hand:
  def __init__(self): 
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_hands = mp.solutions.hands

  def estimate(self,intImg):
    """
    Input: Image

    Return: Marked hand connections
    """
    with self.mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
      # To improve performance, optionally mark the image as not writeable to
      cpImg=intImg.copy()
      image = cv2.cvtColor(intImg, cv2.COLOR_BGR2RGB)
      results = hands.process(image)
      # Draw the hand annotations on the image.
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          self.mp_drawing.draw_landmarks(
              cpImg,
              hand_landmarks,
              self.mp_hands.HAND_CONNECTIONS,
              self.mp_drawing_styles.get_default_hand_landmarks_style(),
              self.mp_drawing_styles.get_default_hand_connections_style())
      return cpImg

# For webcam input:
# Test
if __name__ == "__main__":
    detector=Hand()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if success:
            image=detector.estimate(image)
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()