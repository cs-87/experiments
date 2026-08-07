import cv2


class SIFT:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def get_keypoints(self, y_frame):
        return self.sift.detectAndCompute(y_frame, None)

        