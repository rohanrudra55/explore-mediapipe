import cv2
import mediapipe as mp
import numpy as np

class Bodypose:
  def __init__(self):
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_pose = mp.solutions.pose
  
  def estimate(self,intImg):
    """
    Input:Input image with body

    Return:Estimated poses 
    """
    with self.mp_pose.Pose(
      # static_image_mode=True,
      # model_complexity=2,
      enable_segmentation=False,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
        cpImg=intImg.copy()
        image = cv2.cvtColor(intImg, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        # Draw the pose annotation on the image.
        if results.pose_landmarks:
          # condition = np.stack((results.segmentation_mask) * 3, axis=-1) > 0.1
          # bg_image = np.zeros(image.shape, dtype=np.uint8)
          # bg_image[:] = (192, 192, 192) # gray
          # annotated_image = np.where(condition, cpImg, bg_image)
          self.mp_drawing.draw_landmarks(
              cpImg,
              results.pose_landmarks,
              self.mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return cpImg


# For webcam input:
# Test
if __name__ == "__main__":
    detector=Bodypose()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if success:
            image=detector.estimate(image)
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()