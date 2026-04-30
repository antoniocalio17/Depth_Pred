import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")  # avoid opening windows
import matplotlib.pyplot as plt
from get_keypoints import Keypoints
from sam2.sam2_image_predictor import SAM2ImagePredictor
from datetime import datetime

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    mask = np.asarray(mask)
    if mask.ndim == 3:  # (1,H,W) -> (H,W)
        mask = mask[0]
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

"""
This class is used to predict the mask of the body in the image.

It calls the class Keypoints to get the skeleton.
The skeleton is then passed to SAM2 which gives the mask.


"""

class Mask:
    def __init__(
        self,
        sam_model_id: str = "facebook/sam2-hiera-large",
        device: str | None = None,
        backend: str = "onnxruntime",
        mode: str = "performance",
        openpose_skeleton: bool = False
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.sam_model_id = sam_model_id

        # Keypoints detector
        self.kp_detector = Keypoints(self.device, backend, mode, openpose_skeleton)
        
        # indices in which we are interested in the skeleton output 
        self._default_body_indices = list = [0, 5, 6, 11, 12, 13, 14]

        # SAM2 predictor
        self.sam_predictor = SAM2ImagePredictor.from_pretrained(self.sam_model_id, device=self.device)
        
        # runtime state
        self.image = None
        self.coordinates = None
        self.point_labels = None
        self._mask = None

    def load_image(self, image_path: str):
        """
        Load the image in a format compatible with both the keypoints detector and the SAM2 model.
        In the Keypoints class, this is implemented as cv2.imread(path).
        """
        img = self.kp_detector.load_image(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.image = img
        self.image_path = image_path
        return img

    def set_keypoints_from_pose(self, body_indices: list[int] | None = None):
        """
        Get the keypoints calling the get_keypoints method from the Keypoints class.

        """
        kp_all, _scores = self.kp_detector.get_keypoints(self.image)

        indices = self._default_body_indices

        kpt = np.asarray(kp_all[indices], dtype=np.float32)  # (K,2)
        self.coordinates = kpt.tolist()
        self.point_labels = [1] * len(self.coordinates) # sam needs labels to assign to each pixel of the image we are saying that the pixels belonging to the body are labeled as 1
        return self.coordinates

    def predict_mask(self):
        if self.image is None or self.coordinates is None:
            raise RuntimeError("Call load_image() and set_keypoints_from_pose() before predict_mask().")

        with torch.inference_mode():
            self.sam_predictor.set_image(self.image)
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=np.array(self.coordinates, dtype=np.float32),
                point_labels=np.array(self.point_labels, dtype=np.int32),
            )

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        # normalizza a float 0/1
        if mask.dtype != np.bool_:
            mask = (mask > 0.5)
        self._mask = mask.astype(np.float32)
        return self._mask

    def save_overlay(self, output_path: str, figsize=(20, 10)):
        if self._mask is None:
            raise RuntimeError("Call predict_mask() before save_overlay().")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        show_mask(self._mask, ax=ax, random_color=False)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        

    def mask(self, image_path: str, body_indices: list[int] | None = None):
        self.load_image(image_path)
        self.set_keypoints_from_pose(body_indices=body_indices)
        self.predict_mask()
        return self._mask
    
    def plot_mask(self,output_path: str):
        return self.save_overlay(output_path)


    @staticmethod
    def get_mask_coordinates(mask):
        coordinates = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] > 0.5:
                    coordinates.append((i, j))
        return coordinates

from datetime import datetime

predictor = Mask()
mask = predictor.mask(image_path="/Users/user/Desktop/Thesis/images/penalty.png")
predictor.plot_mask(output_path=f"/Users/user/Desktop/Thesis/DA-V2/Depth-Anything-V2/output/mask_penalty{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")