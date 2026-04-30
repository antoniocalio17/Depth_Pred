import cv2
import torch
from rtmlib import BodyWithFeet

class Keypoints:
    def __init__(self,device, backend,mode,openpose_skeleton = False):
        self.device = device
        self.backend = backend
        self._predictor  = BodyWithFeet(
            to_openpose=openpose_skeleton,
            mode=mode,
            backend=backend,
            device=device,
        )

    def load_image(self, image_path):
        return cv2.imread(image_path)
    
    def predict(self, image):
        keypoints, scores = self._predictor(image)
        return keypoints, scores
    
    def get_keypoints(self,image):
        keypoints, scores = self.predict(image)
        keypoints = keypoints[0].squeeze()   # shape (26, 2) o (N_kp, 2) per HALPE-26
        scores = scores[0].squeeze()        # confidence per keypoint
        return keypoints, scores