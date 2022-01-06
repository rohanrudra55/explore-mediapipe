import cv2
import mediapipe as mp

class Facemesh:
    def __init__(self,max_num_faces=1,refine_landmarks=True,
            min_detection_confidence=0.5,min_tracking_confidence=0.5):

        self.draw = mp.solutions.drawing_utils
        self.drawingStyles = mp.solutions.drawing_styles
        self.drawingSpec = self.draw.DrawingSpec(thickness=1, circle_radius=1)
        self.facemesh = mp.solutions.face_mesh
        self.maxF=max_num_faces
        self.refL=refine_landmarks
        self.mdc=min_detection_confidence
        self.mtc=min_tracking_confidence

    def estimate(self,inImg):
        """
        Input:
            Image with faces

        Return:
            Face mesh with tracked iris , contour

        """
        with self.facemesh.FaceMesh(
            max_num_faces=self.maxF,
            refine_landmarks=self.refL,
            min_detection_confidence=self.mdc,
            min_tracking_confidence=self.mtc) as mesh:

            cpImg=inImg.copy()
            image = cv2.cvtColor(inImg, cv2.COLOR_BGR2RGB)
            results = mesh.process(image)

            # Draw the face mesh annotations on the image.
            if results.multi_face_landmarks:
                for faceLandmarks in results.multi_face_landmarks:
                    self.draw.draw_landmarks(
                        image=cpImg,
                        landmark_list=faceLandmarks,
                        connections=self.facemesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawingStyles
                        .get_default_face_mesh_tesselation_style())

                    # self.draw.draw_landmarks(
                    #     image=cpImg,
                    #     landmark_list=faceLandmarks,
                    #     connections=self.facemesh.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=self.drawingStyles
                    #     .get_default_face_mesh_contours_style())
                    
                    # self.draw.draw_landmarks(
                    #     image=cpImg,
                    #     landmark_list=faceLandmarks,
                    #     connections=self.facemesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=self.drawingStyles
                    #     .get_default_face_mesh_iris_connections_style())
            return cpImg

# For webcam input:
# Test
if __name__ == "__main__":
    detector=Facemesh()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if success:
            image=detector.estimate(image)
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()