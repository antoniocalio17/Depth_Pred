import torchvision.transforms.v2 as v2
import cv2
import torch
import numpy as np

image_path = r"/Users/user/Desktop/Depth Model/segmentation/images/hq_example.png"
model_path = r"/Users/user/Desktop/Depth Model/segmentation/best_model.pth"
best_model = torch.load(model_path)

eval_image_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])




tmp_image_rgb = cv2.imread(image_path)
tmp_image_rgb = cv2.cvtColor(tmp_image_rgb, cv2.COLOR_BGR2RGB)
original_size = tmp_image_rgb.shape[:2]

tmp_input_tensor = torch.from_numpy(tmp_image_rgb).permute(2, 0, 1).float() / 255.0
tmp_input_tensor = eval_image_transform(tmp_input_tensor).unsqueeze(0)

with torch.no_grad():
    output = best_model(tmp_input_tensor)
    output = torch.sigmoid(output).squeeze().cpu().numpy()
    predicted_mask = (output > 0.9).astype(np.uint8) * 255

predicted_mask = cv2.resize(predicted_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

 