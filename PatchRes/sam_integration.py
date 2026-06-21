"""
SAM (Segment Anything Model) integration module.

Provides automatic mask generation for candidate region extraction.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import cv2
import os
import logging

_logger = logging.getLogger(__name__)
_sam_amg_skip_nms_empty_crop_count = 0

_SAM_AUTOMATIC_MASK_GENERATOR_KWARGS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "crop_n_layers": 1,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 1000,
    "output_mode": "binary_mask",
}


def _patch_sam_automatic_mask_generator_generate_masks() -> None:
    """
    Patch for segment-anything#614: crash when crop boxes are empty after filtering.
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator
    except ImportError:
        return
    if getattr(SamAutomaticMaskGenerator._generate_masks, "_res_sam_patched", False):
        return

    from torchvision.ops.boxes import batched_nms, box_area
    from segment_anything.utils.amg import MaskData, generate_crop_boxes

    def _generate_masks_fixed(self, image: np.ndarray) -> Any:
        global _sam_amg_skip_nms_empty_crop_count
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        if len(crop_boxes) > 1:
            if len(data["crop_boxes"]) > 0:
                scores = 1 / box_area(data["crop_boxes"])
                scores = scores.to(data["boxes"].device)
                keep_by_nms = batched_nms(
                    data["boxes"].float(), scores,
                    torch.zeros_like(data["boxes"][:, 0]),
                    iou_threshold=self.crop_nms_thresh,
                )
                data.filter(keep_by_nms)
            else:
                _sam_amg_skip_nms_empty_crop_count += 1

        data.to_numpy()
        return data

    _generate_masks_fixed._res_sam_patched = True
    SamAutomaticMaskGenerator._generate_masks = _generate_masks_fixed


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Convert input image to uint8 grayscale."""
    if image.dtype == np.uint8:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[-1] == 1:
            return image[:, :, 0]

    if image.ndim == 3:
        if image.dtype == np.uint8 and image.shape[-1] >= 3:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(image[:, :, :3].astype(np.float32), cv2.COLOR_RGB2GRAY)

    img = image.astype(np.float32, copy=False)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < 1e-8:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert input image to uint8 RGB."""
    if image.ndim == 3 and image.dtype == np.uint8 and image.shape[-1] == 3:
        return image
    gray_u8 = _to_uint8_gray(image)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)


class SAMIntegration:
    """SAM model integration for automatic mask generation."""

    def __init__(
        self,
        model_type: str = "vit_l",
        checkpoint_path: str = None,
        device: str = "cuda",
    ):
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self._sam = None
        self._mask_generator = None

    def _load_sam(self):
        """Lazy-load SAM model on first use."""
        if self._sam is not None:
            return

        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "segment-anything is not installed. Run: pip install segment-anything"
            )

        _patch_sam_automatic_mask_generator_generate_masks()

        if self.checkpoint_path is None:
            default_paths = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth",
            }
            possible_dirs = [
                os.path.expanduser("~/.cache/torch/hub/checkpoints/"),
                os.path.expanduser("~/models/"),
                "./models/",
                "./",
            ]
            for dir_path in possible_dirs:
                path = os.path.join(dir_path, default_paths.get(self.model_type, "sam_vit_l_0b3195.pth"))
                if os.path.exists(path):
                    self.checkpoint_path = path
                    break

            if self.checkpoint_path is None:
                raise FileNotFoundError(
                    f"SAM checkpoint not found for {self.model_type}. "
                    f"Download from https://dl.fbaipublicfiles.com/segment_anything/..."
                )

        print(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        self._sam = sam_model_registry[self.model_type](checkpoint=None)
        try:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        self._sam.load_state_dict(state_dict, strict=True)
        self._sam.to(device=self.device)

        self._mask_generator = SamAutomaticMaskGenerator(
            self._sam, **_SAM_AUTOMATIC_MASK_GENERATOR_KWARGS,
        )

        print(f"SAM model loaded successfully on {self.device}")

    def generate_masks_automatic(
        self,
        image: np.ndarray,
        min_area_ratio: Optional[float] = None,
        max_area_ratio: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Generate candidate masks in fully automatic mode."""
        self._load_sam()
        image = _to_uint8_rgb(image)

        masks = self._mask_generator.generate(image)
        masks = [m for m in masks if m.get("area", 0) < 2e4]

        img_area = image.shape[0] * image.shape[1]
        min_area = int(img_area * min_area_ratio) if min_area_ratio is not None else None
        max_area = int(img_area * max_area_ratio) if max_area_ratio is not None else None

        candidate_regions = []
        for mask_info in masks:
            area = mask_info['area']
            area_ok = True
            if min_area is not None:
                area_ok = area_ok and (area >= min_area)
            if max_area is not None:
                area_ok = area_ok and (area <= max_area)
            if area_ok:
                x, y, w, h = mask_info['bbox']
                bbox = [int(x), int(y), int(x + w), int(y + h)]
                candidate_regions.append({
                    'bbox': bbox,
                    'mask': mask_info['segmentation'],
                    'area': area,
                    'stability_score': mask_info['stability_score'],
                    'predicted_iou': mask_info['predicted_iou'],
                })

        candidate_regions.sort(key=lambda x: x['stability_score'], reverse=True)
        return candidate_regions

    def extract_region(self, image: np.ndarray, bbox: List[int], padding: int = 5) -> np.ndarray:
        """Extract a region from the image given a bounding box."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return image[y1:y2, x1:x2]
