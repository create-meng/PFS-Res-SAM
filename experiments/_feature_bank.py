"""
Res-SAM current mainline - Step 1: build the feature bank.

Fixed configuration:
- num_normal_samples = 20
- coreset_ratio = 0.5
- grouped_sampling = True
- use_rotation_aug = 0
"""

import sys
import os
import json
import hashlib
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from experiments.resize_policy import RESIZE_POLICY_FIXED, target_hw_for_preprocess
from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_01
from experiments.paper_constants import DEFAULT_BETA_THRESHOLD, preflight_faiss_or_raise
from PatchRes.logger import setup_global_logger, log_config, log_finish


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


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "normal_data_sources": {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact"),
    },
    "anomaly_data_sources": {
        "augmented_cavities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities"),
        "augmented_utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities"),
    },
    "output_dir": os.path.join(BASE_DIR, "outputs", "feature_banks"),
    "output_file": "feature_bank.pth",
    "metadata_file": "metadata.json",
    "bank_suffix": _normalize_suffix("BANK_SUFFIX", ""),
    
    # Core parameters
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    "num_normal_samples": 20,
    "grouped_sampling": True,
    
    # Fixed: no rotation augmentation
    "use_rotation_aug": 0,
    "rotation_angles": [],
    
    # Coreset
    "use_coreset": True,
    "coreset_ratio": 0.5,
    "coreset_min_size": 500,
    "coreset_max_size": 5000,
    
    # GPR background removal
    "gpr_background_removal": True,
    "background_removal_method": "both",
    
    # Validation set
    "num_val_normal": 20,
    "num_val_anomaly": 20,
    "val_seed": 42,
    
    # Other
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "device": "auto",
    "random_seed": 11,
    "version": "current",
    "feature_with_bias": True,
    "use_softpatch_bank_filter": 0,
    "softpatch_keep_ratio": float(os.getenv("SOFTPATCH_KEEP_RATIO", "0.85")),
    "softpatch_knn_k": int(os.getenv("SOFTPATCH_KNN_K", "5")),
    "softpatch_score_power": float(os.getenv("SOFTPATCH_SCORE_POWER", "1.0")),
}


def _to_abs(base_dir: str, p: str) -> str:
    if not p: return p
    if os.path.isabs(p): return p
    return os.path.abspath(os.path.join(base_dir, p))


def get_image_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def remove_gpr_background(img_array: np.ndarray, method: str = "both") -> np.ndarray:
    """GPR B-scan background removal."""
    result = img_array.copy().astype(np.float32)
    if method in ("row_mean", "both"):
        result = result - result.mean(axis=1, keepdims=True)
    if method in ("col_mean", "both"):
        result = result - result.mean(axis=0, keepdims=True)
    std = result.std()
    if std > 1e-8:
        result = result / std
    return result.astype(np.float32)


def _get_original_id(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    m = re.match(r'^(.+?)_aug_\d+$', name)
    return m.group(1) if m else name


def split_validation_set(config: dict) -> dict:
    """Split validation set."""
    print("\n=== Split validation set ===")
    rng = np.random.RandomState(config["val_seed"])
    
    # Normal validation set
    normal_dir = config["normal_data_sources"]["augmented_intact"]
    all_normal = sorted([f for f in os.listdir(normal_dir)
                         if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    
    val_normal_indices = rng.choice(len(all_normal), size=config["num_val_normal"], replace=False)
    val_normal_files = [all_normal[i] for i in sorted(val_normal_indices)]
    train_normal_files = [f for f in all_normal if f not in val_normal_files]
    
    print(f"  Normal: validation {len(val_normal_files)} images, training {len(train_normal_files)} images")
    
    # Anomaly validation set
    val_anomaly_files = []
    for source_name, source_dir in config["anomaly_data_sources"].items():
        all_anomaly = sorted([f for f in os.listdir(source_dir)
                              if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        num_from_this = config["num_val_anomaly"] // len(config["anomaly_data_sources"])
        indices = rng.choice(len(all_anomaly), size=min(num_from_this, len(all_anomaly)), replace=False)
        selected = [(source_name, all_anomaly[i]) for i in indices]
        val_anomaly_files.extend(selected)
    
    print(f"  Anomaly: {len(val_anomaly_files)} images")
    
    return {
        'val_normal_files': val_normal_files,
        'val_anomaly_files': val_anomaly_files,
        'train_normal_files': train_normal_files,
    }


def apply_rotation_augmentation(img: np.ndarray, angles: list) -> list:
    """Rotation augmentation: generate rotated variants."""
    from scipy.ndimage import rotate
    augmented = [img]  # original
    for angle in angles:
        rotated = rotate(img, angle, reshape=False, order=1, mode='constant', cval=0)
        augmented.append(rotated)
    return augmented


def load_images(file_list, base_dir, config: dict) -> tuple:
    """Load images from file list."""
    target_hw = target_hw_for_preprocess(config, None)
    images, paths, hashes = [], [], []
    
    use_rotation = config.get("use_rotation_aug", 0)
    rotation_angles = config.get("rotation_angles", [90, 180, 270])
    
    for img_file in tqdm(file_list, desc="  Loading"):
        img_path = os.path.join(base_dir, img_file)
        try:
            img = Image.open(img_path).convert("L")
            if target_hw:
                img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            if config.get("gpr_background_removal", False):
                arr = remove_gpr_background(arr, config.get("background_removal_method", "both"))
            
            # Rotation augmentation
            if use_rotation:
                augmented_images = apply_rotation_augmentation(arr, rotation_angles)
                for aug_img in augmented_images:
                    images.append(aug_img)
                    paths.append(img_path)
                    hashes.append(get_image_hash(img_path))
            else:
                images.append(arr)
                paths.append(img_path)
                hashes.append(get_image_hash(img_path))
        except Exception as e:
            print(f"  Warning: failed to load {img_file}: {e}")
    
    return images, paths, hashes


def load_normal_images_for_training(train_files, data_dir, config: dict) -> tuple:
    """Sample 20 normal images for feature bank construction."""
    num_samples = config.get("num_normal_samples", 20)
    rng = np.random.RandomState(config.get("random_seed", 11))
    
    if config.get("grouped_sampling", False):
        groups: dict = {}
        for f in train_files:
            gid = _get_original_id(f)
            groups.setdefault(gid, []).append(f)
        group_ids = sorted(groups.keys())
        selected = [rng.choice(groups[gid]) for gid in group_ids]
        if len(selected) < num_samples:
            remaining = [f for f in train_files if f not in selected]
            rng.shuffle(remaining)
            selected += remaining[:num_samples - len(selected)]
        selected = selected[:num_samples]
        print(f"  Grouped sampling: {len(group_ids)} original groups -> selected {len(selected)} images")
    else:
        indices = rng.choice(len(train_files), size=min(num_samples, len(train_files)), replace=False)
        selected = [train_files[i] for i in sorted(indices)]
    
    return load_images(selected, data_dir, config)


def greedy_coreset_sampling(features: np.ndarray, target_size: int, seed: int = 11) -> np.ndarray:
    """PatchCore coreset sampling."""
    n = features.shape[0]
    if target_size >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(0, n))]
    min_dists = np.full(n, np.inf, dtype=np.float32)
    print(f"  Coreset: {n} -> {target_size} patches...")
    for _ in tqdm(range(1, target_size), desc="  Coreset"):
        last = features[selected[-1]]
        dists = np.sum((features - last) ** 2, axis=1).astype(np.float32)
        min_dists = np.minimum(min_dists, dists)
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        min_dists[next_idx] = 0.0
    return np.array(selected, dtype=np.int64)


def apply_softpatch_bank_filter(features: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    """Approximate SoftPatch: remove isolated normal patches, then coreset."""
    keep_ratio = float(config.get("softpatch_keep_ratio", 1.0))
    knn_k = max(2, int(config.get("softpatch_knn_k", 5)))
    score_power = max(0.1, float(config.get("softpatch_score_power", 1.0)))

    if features.shape[0] <= knn_k + 1 or keep_ratio >= 0.999:
        stats = {
            "enabled": True,
            "applied": False,
            "keep_ratio": keep_ratio,
            "knn_k": knn_k,
            "score_power": score_power,
            "num_before": int(features.shape[0]),
            "num_after": int(features.shape[0]),
        }
        return features, stats

    import faiss

    xb = np.ascontiguousarray(features.astype(np.float32, copy=False))
    index = faiss.IndexFlatL2(int(xb.shape[1]))
    index.add(xb)
    dists, _ = index.search(xb, knn_k + 1)
    local_density = np.sqrt(np.maximum(dists[:, 1:], 0.0)).mean(axis=1)
    weighted_density = np.power(local_density, score_power)

    keep_count = max(knn_k + 1, int(round(features.shape[0] * keep_ratio)))
    keep_indices = np.argsort(weighted_density)[:keep_count]
    filtered = features[np.sort(keep_indices)]
    stats = {
        "enabled": True,
        "applied": True,
        "keep_ratio": keep_ratio,
        "knn_k": knn_k,
        "score_power": score_power,
        "num_before": int(features.shape[0]),
        "num_after": int(filtered.shape[0]),
        "local_density_mean": float(local_density.mean()),
        "local_density_p90": float(np.percentile(local_density, 90)),
        "local_density_max": float(local_density.max()),
    }
    return filtered, stats


def calibrate_beta_with_validation(model, feature_bank_np, val_split, config: dict) -> dict:
    """Validation-set-driven beta calibration."""
    print("\n=== Validation-set-driven Beta calibration ===")
    
    # Load validation set
    normal_dir = config["normal_data_sources"]["augmented_intact"]
    val_normal_images, _, _ = load_images(val_split['val_normal_files'], normal_dir, config)
    
    val_anomaly_images = []
    for source_name, img_file in val_split['val_anomaly_files']:
        source_dir = config["anomaly_data_sources"][source_name]
        imgs, _, _ = load_images([img_file], source_dir, config)
        val_anomaly_images.extend(imgs)
    
    print(f"  Validation: {len(val_normal_images)} normal + {len(val_anomaly_images)} anomaly")
    
    # Compute scores
    print("  Computing normal validation scores...")
    normal_scores = []
    for img in tqdm(val_normal_images, desc="  Normal"):
        patches = model._extract_patches(img)
        if patches.shape[0] == 0:
            continue
        features = model._fit_patches(patches).detach().cpu().numpy()
        scores = model._score_features_against_bank(features)
        normal_scores.extend(scores.tolist())
    
    print("  Computing anomaly validation scores...")
    anomaly_scores = []
    for img in tqdm(val_anomaly_images, desc="  Anomaly"):
        patches = model._extract_patches(img)
        if patches.shape[0] == 0:
            continue
        features = model._fit_patches(patches).detach().cpu().numpy()
        scores = model._score_features_against_bank(features)
        anomaly_scores.extend(scores.tolist())
    
    normal_scores = np.array(normal_scores)
    anomaly_scores = np.array(anomaly_scores)
    
    print(f"\n  Patch-level scores:")
    print(f"    Normal: n={len(normal_scores)}, mean={normal_scores.mean():.4f}, "
          f"p90={np.percentile(normal_scores,90):.4f}, max={normal_scores.max():.4f}")
    print(f"    Anomaly: n={len(anomaly_scores)}, mean={anomaly_scores.mean():.4f}, "
          f"p10={np.percentile(anomaly_scores,10):.4f}, max={anomaly_scores.max():.4f}")
    
    # Compute candidate beta
    beta_candidates = {
        'anomaly_p10': float(np.percentile(anomaly_scores, 10)),
        'anomaly_p15': float(np.percentile(anomaly_scores, 15)),
        'anomaly_p20': float(np.percentile(anomaly_scores, 20)),
        'normal_p90': float(np.percentile(normal_scores, 90)),
        'normal_p95': float(np.percentile(normal_scores, 95)),
    }
    
    final_beta = beta_candidates['anomaly_p10']
    print(f"\n  Final beta = {final_beta:.4f}  (strategy: anomaly_p10)")
    
    return {
        'beta': final_beta,
        'beta_method': 'validation_set_anomaly_p10',
        'beta_candidates': beta_candidates,
        'val_stats': {
            'normal_scores': {
                'n': int(len(normal_scores)),
                'mean': float(normal_scores.mean()),
                'std': float(normal_scores.std()),
                'max': float(normal_scores.max()),
            },
            'anomaly_scores': {
                'n': int(len(anomaly_scores)),
                'mean': float(anomaly_scores.mean()),
                'std': float(anomaly_scores.std()),
                'p10': float(np.percentile(anomaly_scores, 10)),
                'max': float(anomaly_scores.max()),
            },
        },
    }


def build_feature_bank(config: dict):
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    print("=" * 60)
    print("Res-SAM: Feature Bank construction")
    print("=" * 60)
    print(f"  hidden_size              = {config['hidden_size']}")
    print(f"  background_removal       = {config.get('background_removal_method')}")
    print("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], config["output_file"])

    from PatchRes.ResSAM import ResSAM

    # 1. Split validation set
    val_split = split_validation_set(config)
    
    # 2. Initialize model
    print("\nInitializing ResSAM...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        beta_threshold=float(config.get("beta_threshold", DEFAULT_BETA_THRESHOLD)),
        feature_with_bias=bool(config.get("feature_with_bias", True)),
        device=config.get("device", "auto"),
    )

    # 3. Build feature bank from training set
    print("\nBuilding Feature Bank from training set...")
    normal_dir = config["normal_data_sources"]["augmented_intact"]
    images, paths, hashes = load_normal_images_for_training(
        val_split['train_normal_files'], normal_dir, config
    )
    
    all_features_list = []
    print(f"\nExtracting patch features from {len(images)} images...")
    for img in tqdm(images, desc="  Extracting"):
        patches = model._extract_patches(img)
        if patches.shape[0] == 0:
            continue
        feats = model._fit_patches(patches)
        all_features_list.append(feats.detach().cpu().numpy())

    if not all_features_list:
        raise RuntimeError("No features extracted")

    all_features = np.concatenate(all_features_list, axis=0).astype(np.float32)
    all_features_before_filter = int(all_features.shape[0])
    print(f"\nFull features: {all_features.shape}")

    softpatch_stats = {"enabled": False, "applied": False}
    if int(config.get("use_softpatch_bank_filter", 0)):
        print("\nApplying SoftPatch-style bank denoising...")
        all_features, softpatch_stats = apply_softpatch_bank_filter(all_features, config)
        print(
            "  SoftPatch filter: "
            f"{softpatch_stats['num_before']} -> {softpatch_stats['num_after']} patches"
        )

    # 4. Coreset sampling
    target_size = max(
        config["coreset_min_size"],
        min(config["coreset_max_size"],
            int(all_features.shape[0] * config["coreset_ratio"]))
    )
    coreset_indices = greedy_coreset_sampling(all_features, target_size, seed=config["random_seed"])
    coreset_features = all_features[coreset_indices]
    print(f"Coreset features: {coreset_features.shape}")

    # 5. Initialize Feature Bank
    model.feature_bank = torch.from_numpy(coreset_features).to(model.device)
    model.anomaly_scorer.fit([coreset_features])
    model._init_nn_searcher(coreset_features)
    
    # 6. Validation-set-driven beta calibration
    beta_result = calibrate_beta_with_validation(model, coreset_features, val_split, config)
    
    # 7. Save Feature Bank
    model.save_feature_bank(output_path)
    print(f"\nFeature Bank saved to: {output_path}")

    # 8. Save metadata
    all_metadata = {
        "config": {k: v for k, v in config.items() if not callable(v)},
        "creation_time": datetime.now().isoformat(),
        "version": "current",
        "feature_bank_shape": list(coreset_features.shape),
        "feature_bank_path": output_path,
        "total_patches_before_coreset": int(all_features.shape[0]),
        "total_patches_before_softpatch_filter": all_features_before_filter,
        "coreset_size": int(coreset_features.shape[0]),
        "softpatch_bank_filter": softpatch_stats,
        "adaptive_beta": beta_result['beta'],
        "beta_calibration_method": beta_result['beta_method'],
        "beta_candidates": beta_result['beta_candidates'],
        "validation_set": {
            "num_normal": len(val_split['val_normal_files']),
            "num_anomaly": len(val_split['val_anomaly_files']),
            "stats": beta_result['val_stats'],
        },
        "sources": {
            "augmented_intact": {
                "directory": normal_dir,
                "num_images": len(images),
                "image_hashes": hashes[:20],
            }
        },
    }

    metadata_path = os.path.join(config["output_dir"], config["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nFinal adaptive_beta = {beta_result['beta']:.4f}")

    return torch.from_numpy(coreset_features)


if __name__ == "__main__":
    logger = setup_global_logger(BASE_DIR, "01_build_feature_bank_current")
    
    preflight_faiss_or_raise()
    CONFIG = apply_layout_to_config_01(dict(CONFIG), BASE_DIR, "current")
    CONFIG["version"] = "current"
    bank_suffix = CONFIG.get("bank_suffix", "")
    if bank_suffix:
        CONFIG["output_dir"] = os.path.join(BASE_DIR, "outputs", f"feature_banks_{bank_suffix}")
        CONFIG["output_file"] = f"feature_bank_{bank_suffix}.pth"
    else:
        CONFIG["output_dir"] = os.path.join(BASE_DIR, "outputs", "feature_banks")
        CONFIG["output_file"] = "feature_bank.pth"
    CONFIG["normal_data_sources"] = {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact")
    }
    CONFIG["anomaly_data_sources"] = {
        "augmented_cavities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities"),
        "augmented_utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities"),
    }
    CONFIG["output_dir"] = _to_abs(BASE_DIR, CONFIG["output_dir"])
    CONFIG["normal_data_sources"] = {k: _to_abs(BASE_DIR, v)
                                      for k, v in CONFIG["normal_data_sources"].items()}
    CONFIG["anomaly_data_sources"] = {k: _to_abs(BASE_DIR, v)
                                       for k, v in CONFIG["anomaly_data_sources"].items()}

    log_config(CONFIG, logger)

    with torch.no_grad():
        build_feature_bank(CONFIG)
    
    log_finish("01_build_feature_bank_current", logger)


