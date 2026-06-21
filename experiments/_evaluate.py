"""
Evaluation script for auto_predictions.json output.
Computes multi-IoU detection metrics and image-level AUC.
"""

from __future__ import print_function
import json
import os
import sys

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

sys.path.insert(0, BASE_DIR)

try:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from PatchRes.logger import setup_global_logger, log_section, log_finish
except ImportError as e:
    print("Error: Missing required packages: numpy, scikit-learn")
    sys.exit(1)


def _normalize_suffix(env_name, fallback=""):
    value = (os.getenv(env_name, fallback) or "").strip()
    if not value:
        return ""
    invalid_chars = set('/\\:')
    if any(ch in value for ch in invalid_chars):
        raise ValueError(f"{env_name} contains illegal path characters: {value!r}")
    return value


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


def get_image_score_strategies(record):
    region_scores = record.get('anomaly_scores', []) or []
    region_max = max(region_scores) if region_scores else 0.0
    patch_max = float(record.get('max_patch_score', 0.0) or 0.0)
    patch_mean = float(record.get('mean_patch_score', 0.0) or 0.0)
    num_patches = int(record.get('num_patches', 0) or 0)
    num_anomaly_patches = int(record.get('num_anomaly_patches', 0) or 0)
    ratio = (float(num_anomaly_patches) / float(num_patches)) if num_patches > 0 else 0.0
    return {
        'region_max': region_max,
        'patch_mean': patch_mean,
        'patch_max': patch_max,
        'blend_region_patch': 0.5 * region_max + 0.5 * patch_max,
        'density_weighted_patch': patch_max * (1.0 + ratio),
    }


def evaluate(pred_path, meta_path):
    """Evaluate prediction results against ground truth annotations."""
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return None

    results = obj.get('results', {})
    pred_meta = obj.get('meta', {})

    metrics_by_iou = {}
    for iou_thresh in [0.1, 0.2, 0.3, 0.5]:
        tp, fp, fn = 0, 0, 0
        tp_scores, fp_scores = [], []

        for cat, recs in results.items():
            if cat == 'normal_auc':
                continue
            for r in recs:
                if r.get('exclude_from_det_metrics'):
                    continue
                pred_bboxes = r.get('pred_bboxes', [])
                gt_bboxes = r.get('gt_bboxes', [])
                scores = r.get('anomaly_scores', [])

                matched_gt = set()
                for pred, score in zip(pred_bboxes, scores):
                    best_iou, best_gt_idx = 0.0, -1
                    for gt_idx, gt in enumerate(gt_bboxes):
                        if gt_idx in matched_gt:
                            continue
                        iou_val = compute_iou(pred, gt)
                        if iou_val > best_iou:
                            best_iou, best_gt_idx = iou_val, gt_idx
                    if best_iou >= iou_thresh:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                        tp_scores.append(score)
                    else:
                        fp += 1
                        fp_scores.append(score)
                fn += len(gt_bboxes) - len(matched_gt)

        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

        metrics_by_iou[iou_thresh] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1,
            'tp_scores': tp_scores, 'fp_scores': fp_scores,
        }

    strategy_scores = {
        'region_max': {'normal': [], 'anomaly': []},
        'patch_mean': {'normal': [], 'anomaly': []},
        'patch_max': {'normal': [], 'anomaly': []},
        'blend_region_patch': {'normal': [], 'anomaly': []},
        'density_weighted_patch': {'normal': [], 'anomaly': []},
    }
    for cat, recs in results.items():
        for r in recs:
            img_scores = get_image_score_strategies(r)
            if cat == 'normal_auc':
                for key in strategy_scores:
                    strategy_scores[key]['normal'].append(img_scores[key])
            elif not r.get('exclude_from_det_metrics'):
                for key in strategy_scores:
                    strategy_scores[key]['anomaly'].append(img_scores[key])

    image_auc_by_strategy = {}
    for key, bucket in strategy_scores.items():
        normal_scores = bucket['normal']
        anomaly_scores = bucket['anomaly']
        y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
        y_score = normal_scores + anomaly_scores
        image_auc_by_strategy[key] = {
            'auc': roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0,
            'normal_mean': float(np.mean(normal_scores)) if normal_scores else 0.0,
            'anomaly_mean': float(np.mean(anomaly_scores)) if anomaly_scores else 0.0,
        }

    total_pred = sum(
        len(r.get('pred_bboxes', []))
        for cat, recs in results.items()
        for r in recs
        if not r.get('exclude_from_det_metrics')
    )

    primary_strategy = 'patch_mean'

    return {
        'meta': meta,
        'pred_meta': pred_meta,
        'metrics_by_iou': metrics_by_iou,
        'primary_image_score_strategy': primary_strategy,
        'image_auc': image_auc_by_strategy[primary_strategy]['auc'],
        'image_auc_by_strategy': image_auc_by_strategy,
        'total_pred': total_pred,
        'normal_mean': image_auc_by_strategy[primary_strategy]['normal_mean'],
        'anomaly_mean': image_auc_by_strategy[primary_strategy]['anomaly_mean'],
    }


if __name__ == '__main__':
    logger = setup_global_logger(BASE_DIR, "evaluate")

    output_suffix = _normalize_suffix("OUTPUT_SUFFIX", "")
    bank_suffix = _normalize_suffix("BANK_SUFFIX", "")
    suffix = output_suffix or ""
    pred_filename = f"auto_predictions{suffix}.json"
    report_filename = f"evaluation_report{suffix}.json"
    pred_path = os.path.join(BASE_DIR, "outputs", "predictions", pred_filename)
    report_path = os.path.join(BASE_DIR, "outputs", "predictions", report_filename)
    if bank_suffix:
        meta_path = os.path.join(BASE_DIR, "outputs", f"feature_banks_{bank_suffix}", "metadata.json")
    else:
        meta_path = os.path.join(BASE_DIR, "outputs", "feature_banks", "metadata.json")

    if not os.path.exists(pred_path):
        logger.error(f"Predictions file not found: {pred_path}")
        logger.error("Run: python experiments/_inference_auto.py first")
        sys.exit(1)

    if not os.path.exists(meta_path):
        logger.error(f"Metadata file not found: {meta_path}")
        logger.error("Run: python experiments/_feature_bank.py first")
        sys.exit(1)

    logger.info(f"Loading predictions: {pred_path}")
    logger.info(f"Loading metadata: {meta_path}")

    result = evaluate(pred_path, meta_path)
    if result is None:
        logger.error("Evaluation failed")
        sys.exit(1)

    m05 = result['metrics_by_iou'][0.5]

    logger.info("Primary metrics (IoU=0.5):")
    logger.info(f"  TP={m05['tp']}, FP={m05['fp']}, F1={m05['f1']:.4f}, Precision={m05['precision']:.4f}")
    logger.info(f"  Image AUC(patch_mean)={result['image_auc_by_strategy']['patch_mean']['auc']:.4f}")
    logger.info(f"  Image AUC(region_max)={result['image_auc_by_strategy']['region_max']['auc']:.4f}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    save_result = {
        'meta': result['meta'],
        'pred_meta': result['pred_meta'],
        'metrics_by_iou': {
            str(k): {
                'tp': v['tp'], 'fp': v['fp'], 'fn': v['fn'],
                'precision': v['precision'], 'recall': v['recall'], 'f1': v['f1'],
            }
            for k, v in result['metrics_by_iou'].items()
        },
        'primary_image_score_strategy': result['primary_image_score_strategy'],
        'image_auc': result['image_auc'],
        'image_auc_by_strategy': result['image_auc_by_strategy'],
        'total_pred': result['total_pred'],
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation report saved: {report_path}")
    log_finish("evaluate", logger)
