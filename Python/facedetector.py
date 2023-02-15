import cv2 as cv
import mediapipe as mp

class Facedetector:
    def __init__(self,model_selection=0,min_detection_confidence=0.5):
        self.faceDetection = mp.solutions.face_detection
        self.drawing = mp.solutions.drawing_utils
        self.modSel=model_selection
        self.minDetCon=min_detection_confidence

    def detect(self,inputImg):
        """
        Input:
            Image containing face / faces

        Returns:
            Image with bounding box around  detected face
            with six spots right eye, left eye, nose tip, 
            mouth center, right ear tragion, and left ear 
            tragion
        """
        with self.faceDetection.FaceDetection(model_selection=self.modSel, min_detection_confidence=self.minDetCon) as facedetect:
            cpImg=inputImg.copy()
            self.image=cv.cvtColor(inputImg, cv.COLOR_BGR2RGB)
            results = facedetect.process(self.image)
            if results.detections:
                for detection in results.detections:
                    # print('Nose tip:',self.faceDetection.get_key_point(detection, self.faceDetection.FaceKeyPoint.NOSE_TIP))
                    self.drawing.draw_detection(cpImg, detection)
            return cpImg
    
    def faces(self,get=True):
        if get:
            return self.count
        else:
            print('Number of faces:',self.count)
        
# For webcam input:
# Test
if __name__ == "__main__":
    detector=Facedetector()
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if success:
            image=detector.detect(image)
            cv.imshow('MediaPipe Face Detection', cv.flip(image, 1))
            if cv.waitKey(5) & 0xFF == 27:
                break
    cap.release()
