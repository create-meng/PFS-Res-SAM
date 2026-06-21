"""
Res-SAM core module: integrates SAM and 2D-ESN for GPR anomaly detection.

Pipeline:
1. SAM generates coarse candidate regions
2. 2D-ESN extracts dynamic features from each region
3. Features compared against normal feature bank
4. Anomaly regions refined and returned as bounding boxes
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union, Sequence
import logging
import os
from PIL import Image

from .ESN_2D_nobatch import ESN_2D
from .common import NearestNeighbourScorer
from .sam_integration import SAMIntegration

FEATURE_BANK_BUNDLE_FORMAT = "res_sam_fb_v2"


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _strict_faiss_default() -> bool:
    """Require faiss unless sklearn fallback is explicitly allowed."""
    if _env_flag("RES_SAM_REQUIRE_FAISS"):
        return True
    if _env_flag("RES_SAM_ALLOW_SKLEARN_KNN"):
        return False
    return True


class ResSAM:
    """
    Res-SAM: GPR anomaly detection combining SAM and Reservoir Computing.

    Parameters:
    -----------
    hidden_size : int
        2D-ESN reservoir size (default 30)
    window_size : int
        Sliding window size (default 50)
    stride : int
        Sliding window stride (default 5)
    spectral_radius : float
        ESN spectral radius (default 0.9)
    connectivity : float
        ESN connectivity (default 0.1)
    beta_threshold : float
        Anomaly threshold beta from Eq.(9)
    sam_model_type : str
        SAM model type ("vit_l", "vit_b", "vit_h")
    sam_checkpoint : str
        Path to SAM checkpoint file
    device : str
        Device for computation ("auto", "cuda", "cpu")
    feature_with_bias : bool
        Include bias term in 2D-ESN readout features
    """

    def __init__(
        self,
        hidden_size: int = 30,
        window_size: int = 50,
        stride: int = 5,
        spectral_radius: float = 0.9,
        connectivity: float = 0.1,
        beta_threshold: float = 0.1,
        sam_model_type: str = "vit_l",
        sam_checkpoint: str = None,
        device: str = "auto",
        *,
        feature_with_bias: bool = True,
    ):
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.stride = stride
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.feature_with_bias = bool(feature_with_bias)
        self.beta_threshold = float(beta_threshold)

        _device_in = (device if device is not None else "").strip().lower() if isinstance(device, str) else device
        if _device_in in {"", None, "auto"}:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif _device_in in {"cuda", "cpu"}:
            self.device = str(_device_in)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _esn_device = torch.device(self.device)
        self.esn = ESN_2D(
            input_dim=1,
            n_reservoir=hidden_size,
            alpha=5,
            spectral_radius=(spectral_radius, spectral_radius),
            connectivity=connectivity,
            device_override=_esn_device,
        )

        self.anomaly_scorer = NearestNeighbourScorer(n_nearest_neighbours=1)
        self.nn_searcher = None
        self.nn_backend = "sklearn"

        self.sam = SAMIntegration(
            model_type=sam_model_type,
            checkpoint_path=sam_checkpoint,
            device=self.device,
        )

        self.feature_bank = None
        self.feature_bank_source = None
        self._beta_calibrated = False

    def _calibrate_beta_threshold(self, feature_bank_np: np.ndarray) -> float:
        """Calibrate beta threshold from feature bank self-nearest-neighbour distances."""
        logger = logging.getLogger(__name__)

        if feature_bank_np is None or (not hasattr(feature_bank_np, "shape")) or len(feature_bank_np.shape) != 2:
            raise ValueError("Invalid feature bank array for beta calibration")
        if int(feature_bank_np.shape[0]) < 3:
            raise ValueError("Feature bank too small for beta calibration")

        quantile = 0.995
        k = 2
        nn_dists = None

        try:
            import faiss
            xb = np.ascontiguousarray(feature_bank_np.astype(np.float32, copy=False))
            index = faiss.IndexFlatL2(int(xb.shape[1]))
            index.add(xb)
            dists, _ = index.search(xb, k)
            nn_dists = np.sqrt(np.maximum(dists[:, 1], 0.0))
        except Exception:
            if _strict_faiss_default():
                raise RuntimeError(
                    "Failed to calibrate beta threshold because faiss is unavailable. "
                    "Set RES_SAM_ALLOW_SKLEARN_KNN=1 to permit sklearn fallback."
                )
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
                nn.fit(feature_bank_np)
                dists, _ = nn.kneighbors(feature_bank_np, n_neighbors=k)
                nn_dists = dists[:, 1]
            except Exception as e:
                raise RuntimeError(f"Failed to calibrate beta threshold (sklearn fallback): {e}")

        nn_dists = np.asarray(nn_dists, dtype=np.float64)
        nn_dists = nn_dists[np.isfinite(nn_dists)]
        if nn_dists.size == 0:
            raise RuntimeError("Beta calibration failed: empty nearest-neighbour distance distribution")

        beta = float(np.quantile(nn_dists, quantile))
        if not np.isfinite(beta):
            raise RuntimeError("Beta calibration failed: non-finite beta")

        self.beta_threshold = beta
        self._beta_calibrated = True

        try:
            logger.info(
                "[BETA] calibrated beta_threshold=%s quantile=%s nn_dist_stats={min:%s,p50:%s,p90:%s,p99:%s,max:%s}",
                beta, quantile,
                float(np.min(nn_dists)), float(np.quantile(nn_dists, 0.5)),
                float(np.quantile(nn_dists, 0.9)), float(np.quantile(nn_dists, 0.99)),
                float(np.max(nn_dists)),
            )
        except Exception:
            pass

        return beta

    def build_feature_bank(
        self,
        normal_images: Union[np.ndarray, Sequence[np.ndarray]],
        source_info: str = "unknown",
    ) -> torch.Tensor:
        """Build feature bank from a set of normal GPR B-scan images."""
        if isinstance(normal_images, np.ndarray):
            arr = normal_images
            if len(arr.shape) == 4:
                arr = arr.squeeze(1)
            iter_images: List[np.ndarray] = [arr[i] for i in range(len(arr))]
        else:
            iter_images = [np.asarray(x) for x in normal_images]

        print(f"Building Feature Bank from {len(iter_images)} normal images...")

        all_features = []
        for img in tqdm(iter_images, desc="Building Feature Bank", unit="img"):
            patches = self._extract_patches(img)
            features = self._fit_patches(patches)
            all_features.append(features)

        self.feature_bank = torch.cat(all_features, dim=0)
        self.feature_bank_source = source_info

        feature_bank_np = self.feature_bank.detach().cpu().numpy()
        self.anomaly_scorer.fit([feature_bank_np])
        self._init_nn_searcher(feature_bank_np)

        print(f"Feature Bank built: shape={self.feature_bank.shape}, source={source_info}")
        return self.feature_bank

    def load_feature_bank(self, path: str):
        """Load a pre-computed feature bank from disk."""
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location="cpu")

        if isinstance(raw, dict) and raw.get("format") == FEATURE_BANK_BUNDLE_FORMAT:
            fb = raw.get("feature_bank")
            if not isinstance(fb, torch.Tensor):
                raise ValueError("Invalid feature bank bundle: missing feature_bank tensor")
            self.feature_bank = fb.to(self.device)
            esd = raw.get("esn_state_dict")
            if isinstance(esd, dict):
                try:
                    self.esn.load_state_dict(esd, strict=True)
                except RuntimeError:
                    logger_fb = logging.getLogger(__name__)
                    msg = "[FeatureBank] ESN state_dict mismatch; falling back to current ESN."
                    print(msg, flush=True)
                    logger_fb.warning(msg)
        elif isinstance(raw, torch.Tensor):
            self.feature_bank = raw.to(self.device)
        else:
            raise ValueError("Unrecognized feature bank file format")

        feature_bank_np = np.ascontiguousarray(self.feature_bank.detach().cpu().numpy(), dtype=np.float32)
        self.anomaly_scorer.fit([feature_bank_np])
        self._init_nn_searcher(feature_bank_np)

        print(f"Feature Bank loaded: shape={self.feature_bank.shape}")

    def _init_nn_searcher(self, feature_bank_np: np.ndarray):
        """Initialize nearest neighbour searcher for patch-level scoring."""
        strict_faiss = _strict_faiss_default()
        backend_env = (os.environ.get("RES_SAM_KNN_BACKEND", "") or "").strip().lower()
        backend = backend_env if backend_env in {"sklearn", "faiss_cpu", "faiss_gpu"} else ""

        if backend == "sklearn":
            if strict_faiss:
                raise RuntimeError(
                    "Strict-faiss default: do not use RES_SAM_KNN_BACKEND=sklearn. "
                    "Set RES_SAM_ALLOW_SKLEARN_KNN=1 if you really want sklearn fallback."
                )
            from sklearn.neighbors import NearestNeighbors
            xb = np.ascontiguousarray(feature_bank_np, dtype=np.float32)
            self.nn_searcher = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1)
            self.nn_searcher.fit(xb)
            self.nn_backend = "sklearn"
            return

        if not backend:
            backend = "faiss_gpu" if (self.device == "cuda" and torch.cuda.is_available()) else "faiss_cpu"

        try:
            import faiss
        except Exception:
            if strict_faiss:
                raise ImportError(
                    "Strict-faiss default but faiss could not be imported. "
                    "Set RES_SAM_ALLOW_SKLEARN_KNN=1 to permit sklearn fallback."
                )
            from sklearn.neighbors import NearestNeighbors
            xb = np.ascontiguousarray(feature_bank_np, dtype=np.float32)
            self.nn_searcher = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1)
            self.nn_searcher.fit(xb)
            self.nn_backend = "sklearn"
            return

        xb = np.ascontiguousarray(feature_bank_np.astype(np.float32, copy=False))
        index = faiss.IndexFlatL2(int(xb.shape[1]))
        if backend == "faiss_gpu":
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                backend = "faiss_cpu"
        index.add(xb)
        self.nn_searcher = index
        self.nn_backend = backend

    def save_feature_bank(self, path: str):
        """Save feature bank with 2D-ESN weights for deterministic inference."""
        bundle = {
            "format": FEATURE_BANK_BUNDLE_FORMAT,
            "feature_bank": self.feature_bank.detach().cpu(),
            "esn_state_dict": {k: v.detach().cpu() for k, v in self.esn.state_dict().items()},
        }
        torch.save(bundle, path)
        print(f"Feature Bank bundle ({FEATURE_BANK_BUNDLE_FORMAT}) saved to {path}")

    def _extract_patches(self, image: np.ndarray) -> torch.Tensor:
        """Sliding window patch extraction on a single image."""
        h, w = image.shape
        win = int(self.window_size)
        stride = int(self.stride)

        if h < win or w < win:
            return torch.zeros((0, 1, win, win), dtype=torch.float32)

        try:
            view = np.lib.stride_tricks.sliding_window_view(image, (win, win))
            patches_np = view[::stride, ::stride].reshape(-1, win, win)
        except Exception:
            patches = []
            for i in range(0, h - win + 1, stride):
                for j in range(0, w - win + 1, stride):
                    patches.append(image[i:i + win, j:j + win])
            patches_np = np.asarray(patches)

        patches_np = np.ascontiguousarray(patches_np, dtype=np.float32)
        return torch.from_numpy(patches_np).unsqueeze(1)

    def _fit_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Fit 2D-ESN on patches and return features (Eq.(2)-(3))."""
        patches_2d = patches.squeeze(1)
        if self.device == "cuda" and torch.cuda.is_available() and patches_2d.device.type != "cuda":
            patches_2d = patches_2d.to(device="cuda", dtype=torch.float32, non_blocking=True)
        elif self.device == "cpu" and patches_2d.device.type != "cpu":
            patches_2d = patches_2d.to(device="cpu", dtype=torch.float32)

        if hasattr(self, "_fitting_count") and isinstance(self._fitting_count, int):
            self._fitting_count += int(patches_2d.shape[0])

        batch_size = 32
        if self.device == "cuda" and torch.cuda.is_available():
            batch_size = 128
            try:
                total_mem_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
                if total_mem_gb >= 14:
                    batch_size = 512
                elif total_mem_gb >= 10:
                    batch_size = 256
            except Exception:
                batch_size = 128
        try:
            env_bs = os.environ.get("RES_SAM_ESN_BATCH", "").strip()
            if env_bs:
                batch_size = max(1, int(env_bs))
        except Exception:
            pass

        feats = []
        with torch.inference_mode():
            for start in range(0, int(patches_2d.shape[0]), batch_size):
                end = min(start + batch_size, int(patches_2d.shape[0]))
                feats.append(self.esn.forward(patches_2d[start:end]))

        if feats:
            features = torch.cat(feats, dim=0)
        else:
            base_dim = 2 * self.hidden_size + 1
            features = torch.zeros((0, base_dim), dtype=torch.float32)

        if (not self.feature_with_bias) and int(features.shape[1]) == (2 * self.hidden_size + 1):
            features = features[:, :-1]
        return features

    def _patch_list_to_tensor(self, patches_list: Union[List[np.ndarray], np.ndarray]) -> torch.Tensor:
        """Convert a list of patch arrays to a batched tensor."""
        if isinstance(patches_list, np.ndarray):
            arr = patches_list
            if arr.size == 0 or arr.shape[0] == 0:
                return torch.zeros((0, 1, int(self.window_size), int(self.window_size)), dtype=torch.float32)
        else:
            if not patches_list:
                return torch.zeros((0, 1, int(self.window_size), int(self.window_size)), dtype=torch.float32)
            arr = np.stack(patches_list, axis=0)

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return torch.from_numpy(arr).unsqueeze(1)

    def _score_features_against_bank(self, features_np: np.ndarray) -> np.ndarray:
        """Score features against the feature bank using nearest-neighbour L2 distance."""
        if self.nn_searcher is None:
            raise RuntimeError(
                "Feature bank searcher not initialized. Call load_feature_bank() or build_feature_bank() first."
            )

        if features_np is None or getattr(features_np, "size", 0) == 0:
            return np.zeros((0,), dtype=np.float32)

        if getattr(self, "nn_backend", "sklearn") == "sklearn":
            distances, _ = self.nn_searcher.kneighbors(features_np)
            return distances.reshape(-1).astype(np.float32, copy=False)

        import numpy as _np
        xq = _np.ascontiguousarray(features_np.astype(_np.float32, copy=False))
        dists, _ = self.nn_searcher.search(xq, 1)
        return _np.sqrt(_np.maximum(dists.reshape(-1), 0.0)).astype(_np.float32, copy=False)

    def _collect_candidate_patches(
        self, image: np.ndarray, bbox: List[int], mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Collect centered patches from a candidate region for fine-grained scoring."""
        img_h, img_w = image.shape
        win = int(self.window_size)
        half_win = win // 2
        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            return np.zeros((0, win, win), dtype=np.float32), []

        y_centers = np.arange(y1, y2, 1, dtype=np.int32)
        x_centers = np.arange(x1, x2, 1, dtype=np.int32)
        if y_centers.size == 0 or x_centers.size == 0:
            return np.zeros((0, win, win), dtype=np.float32), []

        yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)
        y_tops = yy - half_win
        x_lefts = xx - half_win

        valid = (y_tops >= 0) & ((y_tops + win) <= img_h) & (x_lefts >= 0) & ((x_lefts + win) <= img_w)
        if not np.any(valid):
            return np.zeros((0, win, win), dtype=np.float32), []

        y_tops = y_tops[valid]
        x_lefts = x_lefts[valid]
        yy = yy[valid]
        xx = xx[valid]

        view = np.lib.stride_tricks.sliding_window_view(image, (win, win))
        patches_np = np.ascontiguousarray(view[y_tops, x_lefts], dtype=np.float32)
        patch_positions = list(zip(xx.tolist(), yy.tolist()))
        return patches_np, patch_positions

    def _extract_region_feature(self, image: np.ndarray, bbox: List[int], mask: Optional[np.ndarray]) -> torch.Tensor:
        """Fit a SAM coarse region with masked 2D-ESN for coarse-level discrimination."""
        base_dim = 2 * self.hidden_size + (1 if self.feature_with_bias else 0)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return torch.zeros((0, base_dim), dtype=torch.float32)

        crop = np.array(image[y1:y2, x1:x2], dtype=np.float32, copy=True)
        if crop.size == 0:
            return torch.zeros((0, base_dim), dtype=torch.float32)

        if mask is None:
            crop = np.ascontiguousarray(crop, dtype=np.float32)
            crop_tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)
            return self._fit_patches(crop_tensor)

        mask_crop = np.asarray(mask[y1:y2, x1:x2], dtype=bool)
        if mask_crop.shape != crop.shape or (not np.any(mask_crop)):
            return torch.zeros((0, base_dim), dtype=torch.float32)

        crop_tensor = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32)).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_crop, dtype=np.bool_)).unsqueeze(0)
        if self.device == "cuda" and torch.cuda.is_available():
            crop_tensor = crop_tensor.to(device="cuda", dtype=torch.float32, non_blocking=True)
            mask_tensor = mask_tensor.to(device="cuda", non_blocking=True)
        elif self.device == "cpu":
            crop_tensor = crop_tensor.to(device="cpu", dtype=torch.float32)
            mask_tensor = mask_tensor.to(device="cpu")

        if hasattr(self, "_fitting_count") and isinstance(self._fitting_count, int):
            self._fitting_count += 1

        with torch.inference_mode():
            features = self.esn.forward_masked(crop_tensor, mask_tensor)
        if (not self.feature_with_bias) and int(features.shape[1]) == (2 * self.hidden_size + 1):
            features = features[:, :-1]
        return features

    def _merge_overlapping_anomaly_patches(
        self, image_shape: Tuple[int, int], anomaly_positions: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """Merge anomaly patches by overlap connectivity into bounding boxes."""
        if not anomaly_positions:
            return []

        img_h, img_w = image_shape
        half_win = int(self.window_size) // 2
        patch_boxes: List[Tuple[int, int, int, int]] = []
        for cx, cy in anomaly_positions:
            x1 = max(0, int(cx) - half_win)
            y1 = max(0, int(cy) - half_win)
            x2 = min(img_w, int(cx) + half_win)
            y2 = min(img_h, int(cy) + half_win)
            if x2 <= x1 or y2 <= y1:
                continue
            patch_boxes.append((x1, y1, x2, y2))

        if not patch_boxes:
            return []

        def _boxes_overlap(a, b):
            return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

        n = len(patch_boxes)
        visited = [False] * n
        merged_regions: List[Dict[str, Any]] = []

        for start_idx in range(n):
            if visited[start_idx]:
                continue
            stack = [start_idx]
            visited[start_idx] = True
            component_indices: List[int] = []

            while stack:
                idx = stack.pop()
                component_indices.append(idx)
                box_i = patch_boxes[idx]
                for j in range(n):
                    if visited[j]:
                        continue
                    if _boxes_overlap(box_i, patch_boxes[j]):
                        visited[j] = True
                        stack.append(j)

            component_boxes = [patch_boxes[idx] for idx in component_indices]
            x1_all = min(box[0] for box in component_boxes)
            y1_all = min(box[1] for box in component_boxes)
            x2_all = max(box[2] for box in component_boxes)
            y2_all = max(box[3] for box in component_boxes)
            if x2_all <= x1_all or y2_all <= y1_all:
                continue
            merged_regions.append({
                "bbox": [int(x1_all), int(y1_all), int(x2_all), int(y2_all)],
                "num_anomaly_patches": int(len(component_boxes)),
            })

        return merged_regions

    def detect_automatic(
        self,
        image: np.ndarray,
        min_region_area: Optional[int] = None,
        max_regions: Optional[int] = None,
        return_all_candidates: bool = False,
    ) -> Dict[str, Any]:
        """
        Fully automatic anomaly detection.

        1. SAM generates coarse regions on the full B-scan
        2. 2D-ESN fits each coarse region, discards normal-like regions
        3. Retained regions converted to candidate rectangles
        4. Each point in candidate region scored against feature bank
        5. Overlapping anomaly patches merged into final bounding boxes
        """
        if len(image.shape) == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self._fitting_count = 0
        img_h, img_w = image.shape

        print("Step 1: SAM generating coarse regions...")
        coarse_regions = self.sam.generate_masks_automatic(image)
        print(f"  Found {len(coarse_regions)} coarse regions")

        print("Step 2: Region-level coarse filtering...")
        debug_coarse = bool(os.environ.get("RES_SAM_DEBUG_COARSE", "").strip())
        retained_regions = []
        num_discarded = 0

        for region_idx, region in enumerate(coarse_regions):
            bbox = region['bbox']
            mask = region['mask']
            x1, y1, x2, y2 = bbox
            region_area = (x2 - x1) * (y2 - y1)

            if min_region_area is not None and region_area < min_region_area:
                if debug_coarse:
                    print(f"  [coarse][discard][{region_idx}] reason=area_lt_min bbox={bbox}")
                num_discarded += 1
                continue

            region_feature = self._extract_region_feature(image, bbox, mask)
            region_feature_np = region_feature.detach().cpu().numpy()
            region_scores = self._score_features_against_bank(region_feature_np)
            if region_scores.size == 0:
                if debug_coarse:
                    print(f"  [coarse][discard][{region_idx}] reason=no_region_score bbox={bbox}")
                num_discarded += 1
                continue

            region_max_score = float(np.max(region_scores))
            region_mean_score = float(np.mean(region_scores))

            if region_max_score > self.beta_threshold:
                if debug_coarse:
                    print(
                        f"  [coarse][retain][{region_idx}] bbox={bbox} "
                        f"max_score={region_max_score:.6f} mean_score={region_mean_score:.6f}"
                    )
                retained_regions.append({
                    'bbox': bbox,
                    'mask': mask,
                    'coarse_max_score': region_max_score,
                    'coarse_mean_score': region_mean_score,
                    'stability_score': region.get('stability_score', 0),
                })
            else:
                if debug_coarse:
                    print(f"  [coarse][discard][{region_idx}] reason=score_le_beta bbox={bbox}")
                num_discarded += 1

        print(f"  Retained {len(retained_regions)} regions after coarse filtering (discarded {num_discarded})")
        retained_regions.sort(key=lambda r: r['coarse_max_score'], reverse=True)

        print("Step 3: Fine-grained patch analysis...")
        anomaly_regions = []
        regions_for_fine = retained_regions if max_regions is None else retained_regions[:max_regions]

        for region in tqdm(regions_for_fine, desc="  Fine analysis", leave=False):
            bbox = region['bbox']
            mask = region['mask']

            patches_np, patch_positions = self._collect_candidate_patches(image, bbox, mask)
            if patches_np.size == 0 or len(patch_positions) == 0:
                continue

            patches_tensor = self._patch_list_to_tensor(patches_np)
            features = self._fit_patches(patches_tensor)
            features_np = features.detach().cpu().numpy()
            scores = self._score_features_against_bank(features_np)

            if len(patch_positions) == 0:
                continue

            anomaly_positions = [
                pos for pos, s in zip(patch_positions, scores)
                if s > self.beta_threshold
            ]

            if len(anomaly_positions) > 0:
                component_regions = self._merge_overlapping_anomaly_patches((img_h, img_w), anomaly_positions)
                for component in component_regions:
                    final_bbox = component["bbox"]
                    cx1, cy1, cx2, cy2 = final_bbox
                    component_scores = [
                        float(s) for (px, py), s in zip(patch_positions, scores)
                        if (s > self.beta_threshold) and (cx1 <= px < cx2) and (cy1 <= py < cy2)
                    ]
                    if not component_scores:
                        continue
                    anomaly_regions.append({
                        'bbox': final_bbox,
                        'mask': mask,
                        'avg_anomaly_score': float(np.mean(component_scores)),
                        'max_anomaly_score': float(np.max(component_scores)),
                        'coarse_max_score': region['coarse_max_score'],
                        'stability_score': region.get('stability_score', 0),
                        'num_anomaly_patches': int(component['num_anomaly_patches']),
                    })

        anomaly_regions.sort(key=lambda x: x['max_anomaly_score'], reverse=True)
        print(f"  Detected {len(anomaly_regions)} anomaly regions")

        result = {
            'anomaly_regions': anomaly_regions if max_regions is None else anomaly_regions[:max_regions],
            'num_candidates': len(retained_regions),
            'num_coarse_discarded': num_discarded,
            'num_esn_fits': int(getattr(self, "_fitting_count", 0)),
        }
        if return_all_candidates:
            result['all_candidates'] = retained_regions
        return result


def load_image(path: str, size: Tuple[int, int] = None) -> np.ndarray:
    """Load an image from disk as grayscale numpy array."""
    img = Image.open(path).convert('L')
    if size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.array(img)
