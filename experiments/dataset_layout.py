"""
Central dataset paths.

Current policy:
- feature bank source: data/GPR_data/augmented_intact
- eval sets: augmented_cavities / augmented_utilities / augmented_intact
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

DATASET_ENHANCED = "enhanced"
VALID_DATASET_MODES = (DATASET_ENHANCED,)


def resolve_dataset_mode(config: Optional[Dict[str, Any]] = None) -> str:
    return DATASET_ENHANCED


def _voc_anno(subdir: str) -> str:
    return os.path.join(subdir, "annotations", "VOC_XML_format")


def build_layout(
    base_dir: str,
    dataset_mode: str,
    *,
    bank_variant: str = "current",
) -> Dict[str, Any]:
    gpr = os.path.join(base_dir, "data", "GPR_data")

    normal_sources = {
        "augmented_intact": os.path.join(gpr, "augmented_intact"),
    }
    test_data_dirs = {
        "cavities": os.path.join(gpr, "augmented_cavities"),
        "utilities": os.path.join(gpr, "augmented_utilities"),
        "normal_auc": os.path.join(gpr, "augmented_intact"),
    }
    annotation_dirs = {
        "cavities": _voc_anno(os.path.join(gpr, "augmented_cavities")),
        "utilities": _voc_anno(os.path.join(gpr, "augmented_utilities")),
    }

    ver = bank_variant.lower()
    fb_dir = os.path.join(base_dir, "outputs", f"feature_banks_{ver}")
    pred_dir = os.path.join(base_dir, "outputs", f"predictions_{ver}")
    ckpt_dir = os.path.join(base_dir, "outputs", f"checkpoints_{ver}")
    vis_dir = os.path.join(base_dir, "outputs", f"visualizations_{ver}")

    return {
        "dataset_mode": DATASET_ENHANCED,
        "normal_data_sources": normal_sources,
        "test_data_dirs": test_data_dirs,
        "annotation_dirs": annotation_dirs,
        "feature_bank_dir": fb_dir,
        "feature_bank_path": os.path.join(fb_dir, f"feature_bank_{ver}.pth"),
        "metadata_path": os.path.join(fb_dir, "metadata.json"),
        "output_dir_predictions": pred_dir,
        "checkpoint_dir": ckpt_dir,
        "auto_predictions_file": os.path.join(pred_dir, f"auto_predictions_{ver}.json"),
        "vis_output_dir": vis_dir,
        "analysis_output": os.path.join(vis_dir, f"03_evaluate_and_visualize_{ver}_report.md"),
        "metrics_dir": os.path.join(base_dir, "outputs", f"metrics_{ver}"),
        "clustering_output_file": os.path.join(
            base_dir, "outputs", f"metrics_{ver}", f"04_clustering_{ver}.json"
        ),
        "clustering_checkpoint": os.path.join(
            base_dir, "outputs", f"checkpoints_{ver}", f"checkpoint_04_clustering_{ver}.json"
        ),
        "layout_suffix": "",
    }


def apply_layout_to_config_01(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["normal_data_sources"] = lay["normal_data_sources"]
    c["output_dir"] = lay["feature_bank_dir"]
    return c


def apply_layout_to_config_02_03(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["test_data_dirs"] = lay["test_data_dirs"]
    c["annotation_dirs"] = dict(lay["annotation_dirs"])
    c["feature_bank_path"] = lay["feature_bank_path"]
    c["metadata_path"] = lay["metadata_path"]
    c["output_dir"] = lay["output_dir_predictions"]
    c["checkpoint_dir"] = lay["checkpoint_dir"]
    c["annotation_dirs"]["normal_auc"] = ""
    return c


def apply_layout_to_config_04(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["predictions_path"] = lay["auto_predictions_file"]
    c["vis_output_dir"] = lay["vis_output_dir"]
    c["analysis_output"] = lay["analysis_output"]
    return c


def apply_layout_to_config_05(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["predictions_path"] = lay["auto_predictions_file"]
    c["test_data_dirs"] = {k: lay["test_data_dirs"][k] for k in ("cavities", "utilities")}
    c["annotation_dirs"] = {k: lay["annotation_dirs"][k] for k in ("cavities", "utilities")}
    c["output_dir"] = lay["metrics_dir"]
    c["checkpoint_dir"] = lay["checkpoint_dir"]
    return c


def apply_layout_to_config_06(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["test_data_dirs"] = {k: lay["test_data_dirs"][k] for k in ("cavities", "utilities")}
    c["feature_bank_path"] = lay["feature_bank_path"]
    c["output_dir"] = lay["metrics_dir"]
    c["checkpoint_dir"] = lay["checkpoint_dir"]
    return c


def apply_layout_to_config_07(config: Dict[str, Any], base_dir: str, bank_variant: str) -> Dict[str, Any]:
    lay = build_layout(base_dir, resolve_dataset_mode(config), bank_variant=bank_variant)
    c = dict(config)
    c["dataset_mode"] = lay["dataset_mode"]
    c["test_data_dirs"] = lay["test_data_dirs"]
    c["annotation_dirs"] = dict(lay["annotation_dirs"])
    c["annotation_dirs"]["normal_auc"] = ""
    fb = dict(c.get("feature_bank_sources") or {})
    if fb:
        for k in list(fb.keys()):
            fb[k] = lay["feature_bank_path"]
    else:
        fb = {"augmented_intact": lay["feature_bank_path"]}
    c["feature_bank_sources"] = fb
    c["output_dir"] = lay["metrics_dir"]
    c["checkpoint_dir"] = lay["checkpoint_dir"]
    return c
