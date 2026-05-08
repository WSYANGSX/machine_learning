import numpy as np


class ClassificationMetrics:
    """
    Metrics for Image/General Classification Task.
    """

    def __init__(self, nc: int) -> None:
        self.nc = nc
        self.confusion_matrix = np.zeros((nc, nc))

    def update(self, targets: np.ndarray, preds: np.ndarray):
        """
        Update the confusion matrix based on the true labels and predicted labels for a batch of data.
        Args:
            targets: True labels, one-dimensional array, shape as (N,)
            preds:   Predicted labels, one-dimensional array, shape as (N,)
        """
        # Make sure the input is a one-dimensional integer array
        targets = np.asarray(targets).astype(int).flatten()
        preds = np.asarray(preds).astype(int).flatten()

        # Filter out tags that are not within the valid category range
        mask = (targets >= 0) & (targets < self.nc)

        # Quickly calculate the confusion matrix
        hist = np.bincount(
            self.nc * targets[mask] + preds[mask],
            minlength=self.nc**2,
        ).reshape(self.nc, self.nc)

        self.confusion_matrix += hist

    @staticmethod
    def to_str(results):
        string = "\nClassification Metrics:\n" + "-" * 30 + "\n"
        for k, v in results.items():
            # Distinguish between scalar metrics and dictionary-type class-based metrics
            if isinstance(v, dict):
                pass  # If you don't need to print detailed metrics for each category in the console, you can skip this
                # string += f"{k}:\n"
                # for cls_id, val in v.items():
                #     string += f"  Class {cls_id}: {val:.4f}\n"
            else:
                string += f"{k:<16}: {v:.4f}\n"
        string += "-" * 30 + "\n"
        return string

    def get_results(self):
        """
        Return the evaluation result of the classification task:
            - Accuracy (Overall accuracy rate)
            - Macro Precision (macro average precision rate)
            - Macro Recall (Macro average recall rate)
            - Macro F1 (macro average F1-Score
            - Class-wise metrics (metrics of each category)
        """
        cm = self.confusion_matrix

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        total = cm.sum() + 1e-15
        acc = tp.sum() / total

        # Category-by-category indicators
        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        f1 = 2 * precision * recall / (precision + recall + 1e-15)

        # Macro average
        macro_precision = np.nanmean(precision)
        macro_recall = np.nanmean(recall)
        macro_f1 = np.nanmean(f1)

        # Micro-average (equal to accuracy usually in multi-classification)
        micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-15)
        micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-15)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-15)

        # weighted mean
        class_counts = cm.sum(axis=1)
        valid_mask = class_counts > 0
        if valid_mask.any():
            weighted_precision = np.average(precision[valid_mask], weights=class_counts[valid_mask])
            weighted_recall = np.average(recall[valid_mask], weights=class_counts[valid_mask])
            weighted_f1 = np.average(f1[valid_mask], weights=class_counts[valid_mask])
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0

        return {
            "Accuracy": acc,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro F1": macro_f1,
            "Micro Precision": micro_precision,
            "Micro Recall": micro_recall,
            "Micro F1": micro_f1,
            "Weighted Precision": weighted_precision,
            "Weighted Recall": weighted_recall,
            "Weighted F1": weighted_f1,
            "Class Precision": dict(zip(range(self.nc), precision)),
            "Class Recall": dict(zip(range(self.nc), recall)),
            "Class F1": dict(zip(range(self.nc), f1)),
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.nc, self.nc))
