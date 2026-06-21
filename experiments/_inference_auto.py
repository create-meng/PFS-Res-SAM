"""
Res-SAM current mainline - Step 2: fully automatic inference.

Fixed configuration:
- top_k_per_image = 1
- nms_iou_threshold = 0.3
- min_bbox_area = 5000
- hidden_size = 30
- background_removal_method = "both"
- use_multi_scale = 1
- multi_scale_strides = [3, 5, 8]
- multi_scale_weights = [0.33, 0.33, 0.34]
- validation-driven beta calibration
"""

from __future__ import annotations

import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import cv2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_02_03
from experiments.paper_constants import DEFAULT_BETA_THRESHOLD, preflight_faiss_or_raise
from experiments.resize_policy import RESIZE_POLICY_FIXED, target_hw_for_preprocess
from PatchRes.logger import setup_global_logger, log_config, log_section, log_finish

AB_PROGRESS_PREFIX = "__AB_PROGRESS__"


def emit_ablation_progress(kind: str, **kwargs) -> None:
    value = (os.getenv("ABLATION_PROGRESS", "0") or "0").strip().lower()
    if value not in {"1", "true", "yes", "on"}:
        return
    payload = {"kind": kind, **kwargs}
    print(AB_PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _env_flag(name: str, default: str = "0") -> int:
    value = (os.getenv(name, default) or default).strip().lower()
    return 1 if value in {"1", "true", "yes", "on"} else 0


def _normalize_suffix(env_name: str, fallback: str = "") -> str:
    value = (os.getenv(env_name, fallback) or "").strip()
    if not value:
        return ""
    invalid_chars = set('/\\:')
    if any(ch in value for ch in invalid_chars):
        raise ValueError(f"{env_name} contains illegal path characters: {value!r}")
    return value


def _to_abs(base_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks", "feature_bank.pth"),
    "metadata_path": os.path.join(BASE_DIR, "outputs", "feature_banks", "metadata.json"),
    "test_data_dirs": {
        "cavities":   os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities"),
        "utilities":  os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities"),
        "normal_auc": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact"),
    },
    "annotation_dirs": {
        "cavities":  os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities",
                                  "annotations", "VOC_XML_format"),
        "utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities",
                                  "annotations", "VOC_XML_format"),
    },
    "output_dir":     os.path.join(BASE_DIR, "outputs", "predictions"),
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints"),
    "output_suffix": _normalize_suffix("OUTPUT_SUFFIX", ""),
    "bank_suffix": _normalize_suffix("BANK_SUFFIX", ""),

    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    "beta_threshold": DEFAULT_BETA_THRESHOLD,
    "use_adaptive_beta": True,

    "use_score_map": True,
    "score_map_smooth_sigma": 2.0,

    "min_bbox_area": 5000,
    "max_bbox_area": 80000,
    "bbox_expand_pixels": 15,
    "nms_iou_threshold": 0.3,
    "top_k_per_image": 1,

    "use_multi_scale": 1,
    "multi_scale_strides": [3, 5, 8],
    "multi_scale_weights": [0.33, 0.33, 0.34],

    "use_per_image_threshold": True,
    "adaptive_threshold_strategy": "dynamic",
    "per_image_threshold_ratio": 0.7,

    "aspect_ratio_threshold": 2.9,
    "score_threshold_p80": 0.3283,

    "use_secondary_filter": 1,
    "secondary_filter_min_area_orig": 3500,
    "secondary_filter_min_mean_patch": float(os.getenv("SECONDARY_FILTER_MIN_MEAN_PATCH", "0.275")),
    "secondary_filter_box_score_min": float(os.getenv("SECONDARY_FILTER_BOX_SCORE_MIN", "0.0")),

    "gpr_background_removal": True,
    "background_removal_method": "both",

    "device": "auto",
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "max_images_per_category": (
        int(os.getenv("MAX_IMAGES_PER_CATEGORY", "").strip())
        if os.getenv("MAX_IMAGES_PER_CATEGORY", "").strip()
        else None
    ),
    "checkpoint_interval": 50,
    "random_seed": 11,
    "feature_with_bias": True,
}


def remove_gpr_background(arr: np.ndarray, method: str = "both") -> np.ndarray:
    """GPR B-scan background removal via row/column mean subtraction and global normalization."""
    result = arr.copy().astype(np.float32)

    if method in ("row_mean", "both"):
        result = result - result.mean(axis=1, keepdims=True)
    if method in ("col_mean", "both"):
        result = result - result.mean(axis=0, keepdims=True)
    if method == "row_median":
        result = arr - np.median(arr, axis=1, keepdims=True)

    std = result.std()
    if std > 1e-8:
        result = result / std
    return result.astype(np.float32)


def parse_stride_list(raw_value: str | list[int] | tuple[int, ...], fallback: list[int]) -> list[int]:
    if isinstance(raw_value, (list, tuple)):
        values = [int(v) for v in raw_value]
    else:
        text = (raw_value or "").strip()
        if not text:
            values = list(fallback)
        else:
            values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return [v for v in values if v > 0] or list(fallback)


def parse_voc_xml(xml_path: str) -> dict | None:
    """Parse VOC XML annotation to extract bounding box."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None

    obj = root.find('object')
    if obj is None:
        return None
    bndbox = obj.find('bndbox')
    if bndbox is None:
        return None

    try:
        return {
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text),
        }
    except (ValueError, AttributeError):
        return None


def load_image_with_orig_size(path: str, size_hw: tuple[int, int] | None,
                               background_removal: bool = False, removal_method: str = "both") -> tuple:
    """Load image, optionally resize and remove background."""
    img = Image.open(path).convert('L')
    orig_w, orig_h = img.size

    if size_hw:
        h_target, w_target = size_hw
        img = img.resize((w_target, h_target), Image.BILINEAR)
        scale_x = w_target / float(orig_w) if orig_w > 0 else 1.0
        scale_y = h_target / float(orig_h) if orig_h > 0 else 1.0
    else:
        scale_x = scale_y = 1.0

    arr = np.array(img, dtype=np.float32)
    if background_removal:
        arr = remove_gpr_background(arr, method=removal_method)
    else:
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)

    orig_size = (orig_h, orig_w)
    scale_info = (scale_x, scale_y)
    return orig_size, scale_info, arr


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def generate_score_map_single_scale(img_shape: tuple, patch_positions: list,
                                     patch_scores: list, win: int) -> np.ndarray:
    """Generate a dense score map from sparse patch positions and scores via overlap averaging."""
    h, w = img_shape
    score_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    half = win // 2

    for (cx, cy), score in zip(patch_positions, patch_scores):
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        score_map[y1:y2, x1:x2] += score
        count_map[y1:y2, x1:x2] += 1

    np.divide(score_map, count_map, out=score_map, where=count_map > 0)
    return score_map


def build_score_map_from_stride_list(model, image: np.ndarray, strides: list[int],
                                      window_size: int, config: dict, weights: list[float] | None = None) -> tuple:
    """Build multi-stride score map by fusing score maps from multiple patch strides."""
    h, w = image.shape
    maps = []

    for stride in strides:
        model.stride = stride
        patches_np, positions = model._collect_candidate_patches(
            image, [0, 0, w, h], None
        )
        if patches_np.size == 0 or len(positions) == 0:
            continue

        patches_t = model._patch_list_to_tensor(patches_np)
        feats = model._fit_patches(patches_t)
        scores = model._score_features_against_bank(feats.detach().cpu().numpy())

        one_map = generate_score_map_single_scale((h, w), positions, scores.tolist(), window_size)
        maps.append(one_map)

    if not maps:
        return np.zeros((h, w), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

    if weights and len(weights) == len(maps):
        fused = sum(w * m for w, m in zip(weights, maps))
        norm_w = sum(weights)
        if norm_w > 0:
            fused = fused / norm_w
    else:
        fused = np.mean(maps, axis=0)

    smoothed = ndimage.gaussian_filter(fused, sigma=float(config.get("score_map_smooth_sigma", 2.0)))
    return smoothed, np.zeros((0,), dtype=np.float32), []


def nms(bboxes, scores, iou_threshold=0.5):
    """Non-maximum suppression."""
    if len(bboxes) == 0:
        return [], []

    bboxes = np.array(bboxes)
    scores = np.array(scores)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return bboxes[keep].tolist(), scores[keep].tolist()


def detect_with_score_map(model, image: np.ndarray, config: dict,
                           image_id: str = "", logger: logging.Logger | None = None) -> dict:
    """
    Detect anomalies using dense score map + connected components + hierarchical filtering.
    """
    img_h, img_w = image.shape
    win = int(config.get("window_size", 50))
    default_strides = [3, 5, 8]
    default_weights = [0.33, 0.33, 0.34]

    if config.get("use_multi_scale", 1):
        strides = config.get("multi_scale_strides", default_strides)
        weights = config.get("multi_scale_weights", default_weights)
    else:
        strides = [config.get("stride", 5)]
        weights = None

    score_map, _, _ = build_score_map_from_stride_list(
        model, image, strides, win, config, weights
    )

    adaptive_beta = float(config.get("beta_threshold", 0.1))
    if config.get("use_adaptive_beta", False):
        try:
            valid_scores = score_map[score_map > 0]
            if valid_scores.size > 0:
                p80 = float(np.percentile(valid_scores, 80))
                if p80 > adaptive_beta:
                    adaptive_beta = p80
        except Exception:
            pass

    binary = (score_map > adaptive_beta).astype(np.uint8)
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3))).astype(np.uint8)
    binary = ndimage.binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    raw_boxes = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < config.get("min_bbox_area", 5000):
            continue
        x1 = int(stats[i, cv2.CC_STAT_LEFT])
        y1 = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        raw_boxes.append([x1, y1, x1 + w, y1 + h])

    raw_scores = []
    for box in raw_boxes:
        region_scores = score_map[box[1]:box[3], box[0]:box[2]]
        raw_scores.append(float(np.max(region_scores)) if region_scores.size > 0 else 0.0)

    raw_boxes, raw_scores = nms(raw_boxes, raw_scores, float(config.get("nms_iou_threshold", 0.3)))

    top_k = int(config.get("top_k_per_image", 1))
    if top_k > 0 and len(raw_boxes) > top_k:
        paired = sorted(zip(raw_scores, raw_boxes), key=lambda x: x[0], reverse=True)
        raw_boxes = [p[1] for p in paired[:top_k]]
        raw_scores = [p[0] for p in paired[:top_k]]

    final_boxes = []
    final_scores = []
    for box, score in zip(raw_boxes, raw_scores):
        if config.get("use_secondary_filter", 1):
            box_area_orig = (box[2] - box[0]) * (box[3] - box[1])
            min_area = int(config.get("secondary_filter_min_area_orig", 3500))
            if box_area_orig < min_area:
                if logger:
                    logger.debug(f"  [secondary] discard small box area={box_area_orig} < {min_area}")
                continue
            min_mean = float(config.get("secondary_filter_min_mean_patch", 0.275))
            region_scores = score_map[box[1]:box[3], box[0]:box[2]]
            mean_score = float(np.mean(region_scores)) if region_scores.size > 0 else 0.0
            if mean_score < min_mean:
                if logger:
                    logger.debug(f"  [secondary] discard low mean score={mean_score:.4f} < {min_mean}")
                continue
        final_boxes.append(box)
        final_scores.append(score)

    return {
        'boxes': final_boxes,
        'scores': final_scores,
        'score_map': score_map,
        'adaptive_beta': adaptive_beta,
        'num_raw_regions': num_labels - 1 if num_labels > 0 else 0,
    }


def run_inference(config: dict) -> dict:
    """Run fully automatic inference on all test images."""
    from PatchRes.ResSAM import ResSAM

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResSAM(
        hidden_size=int(config.get("hidden_size", 30)),
        window_size=int(config.get("window_size", 50)),
        stride=int(config.get("stride", 5)),
        beta_threshold=float(config.get("beta_threshold", DEFAULT_BETA_THRESHOLD)),
        device=device,
        feature_with_bias=bool(config.get("feature_with_bias", True)),
    )

    fb_path = config.get("feature_bank_path", "")
    if fb_path and os.path.isfile(fb_path):
        print(f"Loading feature bank from {fb_path}")
        model.load_feature_bank(fb_path)
    else:
        raise FileNotFoundError(f"Feature bank not found: {fb_path}")

    predictions = {}
    categories = ["cavities", "utilities", "normal_auc"]

    for cat in categories:
        data_dir = config.get("test_data_dirs", {}).get(cat, "")
        anno_dir = config.get("annotation_dirs", {}).get(cat, "")
        if not data_dir or not os.path.isdir(data_dir):
            print(f"  Skip category '{cat}': data dir not found")
            continue
        print(f"\nProcessing category: {cat}")

        image_files = sorted(
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        max_imgs = config.get("max_images_per_category")
        if max_imgs is not None:
            image_files = image_files[:max_imgs]

        cat_records = []
        for img_name in tqdm(image_files, desc=f"  {cat}", unit="img"):
            img_path = os.path.join(data_dir, img_name)

            try:
                orig_size, scale_info, arr = load_image_with_orig_size(
                    img_path, config.get("image_size"),
                    config.get("gpr_background_removal", True),
                    config.get("background_removal_method", "both"),
                )
            except Exception as e:
                print(f"  Error loading {img_name}: {e}")
                continue

            result = detect_with_score_map(model, arr, config, image_id=img_name,
                                            logger=logging.getLogger(__name__))

            orig_h, orig_w = orig_size
            scale_x, scale_y = scale_info
            boxes_resized = result['boxes']
            boxes_orig = []
            for box in boxes_resized:
                x1 = int(round(box[0] / scale_x)) if scale_x > 0 else box[0]
                y1 = int(round(box[1] / scale_y)) if scale_y > 0 else box[1]
                x2 = int(round(box[2] / scale_x)) if scale_x > 0 else box[2]
                y2 = int(round(box[3] / scale_y)) if scale_y > 0 else box[3]
                boxes_orig.append([x1, y1, x2, y2])

            if cat == "normal_auc":
                record = {
                    'image': img_name,
                    'image_path': img_path,
                    'image_level_score': float(np.mean(result['score_map'])),
                    'max_patch_score': float(np.max(result['score_map'])),
                    'mean_patch_score': float(np.mean(result['score_map'])),
                    'num_patches': int(result['score_map'].size),
                }
            else:
                record = {
                    'image': img_name,
                    'image_path': img_path,
                    'boxes_resized': boxes_resized,
                    'boxes_original': boxes_orig,
                    'scores': result['scores'],
                    'image_level_score': float(np.mean(result['score_map'])),
                    'max_patch_score': float(np.max(result['score_map'])),
                    'mean_patch_score': float(np.mean(result['score_map'])),
                    'adaptive_beta': float(result.get('adaptive_beta', 0)),
                }

            cat_records.append(record)

        predictions[cat] = cat_records

    return predictions


def apply_layout_and_env(custom_cfg: dict | None = None) -> dict:
    """Apply dataset layout and environment variable overrides to config."""
    cfg = dict(CONFIG)
    if custom_cfg:
        cfg.update(custom_cfg)
    cfg = apply_layout_to_config_02_03(cfg, BASE_DIR, bank_variant="current")
    suffix = cfg.get("output_suffix", "")
    bank_suffix = cfg.get("bank_suffix", "")
    if suffix:
        cfg["output_dir"] = cfg["output_dir"] + suffix
    if bank_suffix:
        fb = cfg.get("feature_bank_path", "")
        if fb:
            base, ext = os.path.splitext(fb)
            cfg["feature_bank_path"] = f"{base}{bank_suffix}{ext}"
    return cfg


if __name__ == "__main__":
    preflight_faiss_or_raise()
    cfg = apply_layout_and_env()
    logger = setup_global_logger(BASE_DIR, "inference_auto")

    print("=" * 60)
    print("PFS-Res-SAM: Fully Automatic Inference")
    print("=" * 60)

    log_config(cfg, logger)

    predictions = run_inference(cfg)

    output_dir = cfg.get("output_dir", os.path.join(BASE_DIR, "outputs", "predictions"))
    os.makedirs(output_dir, exist_ok=True)

    meta_path = cfg.get("metadata_path", "")
    meta = {}
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

    output = {
        "meta": {
            "version": "current",
            "timestamp": datetime.now().isoformat(),
            "feature_bank_path": cfg.get("feature_bank_path", ""),
            "config": {k: v for k, v in cfg.items() if not isinstance(v, (dict, list)) or k in ("multi_scale_strides", "multi_scale_weights")},
        },
        "results": predictions,
    }

    output_file = os.path.join(output_dir, "auto_predictions.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nPredictions saved to {output_file}")

    log_finish("inference_auto", logger)
    print("Done.")
