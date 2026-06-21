"""Utility functions for data loading, preprocessing, and mask generation."""

import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def jpg_to_tensor(path: str, to_one_channel=True, size=(369, 369)):
    """Load a JPEG image and convert to a normalized tensor."""
    try:
        image = Image.open(path)
    except IOError:
        print("Error opening image.")
        return None
    transform = transforms.ToTensor()
    if to_one_channel:
        image = image.convert('L')
    tensor_image = transform(image)
    tensor_image = tensor_image.unsqueeze(0)
    if size is not None:
        tensor_image = F.interpolate(tensor_image, size=size, mode='bilinear', align_corners=False)
    tensor_image = tensor_image[0, :, :]
    return tensor_image


def random_select_images_in_one_folder(data_folder: str, num: int, to_one_channel=True,
                                        rand_select=False, size=(369, 369),
                                        return_list=False, return_paths=False):
    """Select images from a folder, convert to tensors, and standardize."""
    all_tensors = []
    images = [os.path.join(data_folder, img)
              for img in os.listdir(data_folder)
              if img.lower().endswith('.jpg') or img.lower().endswith('.png')]
    if rand_select:
        selected_images = random.sample(images, min(num, len(images)))
    else:
        selected_images = images[:min(num, len(images))]

    for image_path in selected_images:
        image_tensor = jpg_to_tensor(image_path, to_one_channel=to_one_channel, size=size)
        image_tensor = (image_tensor - image_tensor.mean()) / image_tensor.std()
        all_tensors.append(image_tensor)

    if return_list:
        return all_tensors
    concatenated_tensor = torch.cat(all_tensors, dim=0) if to_one_channel else torch.stack(all_tensors, dim=0)
    if return_paths:
        return concatenated_tensor, selected_images
    return concatenated_tensor


def generate_masks(images, tile_scores, mode="pixel_seg", window_size=[50, 50],
                   anomaly_threshold=0.3, stride=10, multiple_frames=False,
                   return_box=False, normalization=True, frame_width=5):
    """Generate masks from tile scores. Supports 'OD' (outline) and 'pixel_seg' modes."""
    tile_masks = tile_scores.reshape(
        images.shape[0],
        int((images.shape[1] - window_size[1]) / stride + 1), -1,
    )
    tile_masks = torch.from_numpy(tile_masks)
    tile_masks_t = tile_masks

    masks = F.interpolate(
        tile_masks_t.unsqueeze(1),
        size=[images.shape[-2] - window_size[0], images.shape[-1] - window_size[1]],
        mode="bilinear", align_corners=False,
    ).squeeze(1)
    masks = masks.cpu().numpy()
    if normalization:
        masks = (masks - np.min(masks)) / (np.max(masks) - np.min(masks))

    foreground_pixels = np.argwhere(masks > anomaly_threshold)
    if foreground_pixels.size == 0:
        if return_box:
            return (0, 0, 0, 0)
        extracted_image = None
    else:
        min_row = int(np.min(foreground_pixels[:, 1]))
        min_col = int(np.min(foreground_pixels[:, 2]))
        max_row = int(np.max(foreground_pixels[:, 1]))
        max_col = int(np.max(foreground_pixels[:, 2]))
        extracted_image = images[:, min_row:max_row, min_col:max_col]

    new_masks = np.zeros(images.shape[:])
    if mode == "OD":
        if foreground_pixels.size > 0:
            from sklearn.cluster import DBSCAN
            if multiple_frames:
                clustering = DBSCAN(eps=window_size[0]/stride, min_samples=1).fit(foreground_pixels)
                labels = clustering.fit_predict(foreground_pixels)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                for i in range(n_clusters):
                    one_anomaly_pixels = foreground_pixels[labels == i]
                    min_row = int(np.min(one_anomaly_pixels[:, 1]))
                    min_col = int(np.min(one_anomaly_pixels[:, 2]))
                    max_row = int(np.max(one_anomaly_pixels[:, 1]))
                    max_col = int(np.max(one_anomaly_pixels[:, 2]))
                    masks[:, min_row, min_col:max_col] = 1
                    masks[:, max_row, min_col:max_col] = 1
                    masks[:, min_row:max_row, min_col] = 1
                    masks[:, min_row:max_row, max_col] = 1
                    masks[masks < 1] = 0
            else:
                min_row, min_col, max_row, max_col = (
                    int(min_row + window_size[0]/2), int(min_col + window_size[1]/2),
                    int(max_row + window_size[0]/2), int(max_col + window_size[1]/2)
                )
                if return_box:
                    return (min_col, min_row, max_col, max_row)
                new_masks[:, min_row:min_row+frame_width, min_col:max_col] = 1
                new_masks[:, max_row-frame_width:max_row, min_col:max_col] = 1
                new_masks[:, min_row:max_row, min_col:min_col+frame_width] = 1
                new_masks[:, min_row:max_row, max_col-frame_width:max_col] = 1

    elif mode == "pixel_seg":
        for i in range(masks.shape[-2]):
            for j in range(masks.shape[-1]):
                new_masks[:, i:i+window_size[0], j:j+window_size[1]] += masks[:, i, j]
        if normalization:
            new_masks = (new_masks - new_masks.min()) / (new_masks.max() - new_masks.min())

    return tile_masks, new_masks, extracted_image
