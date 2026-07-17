import torch


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, logits, target):
        prediction = logits.argmax(1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        encoded = target[valid] * self.num_classes + prediction[valid]
        self.confusion += torch.bincount(encoded, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    def compute(self):
        matrix = self.confusion.float()
        intersection = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - intersection
        valid = union > 0
        iou = torch.zeros_like(union)
        iou[valid] = intersection[valid] / union[valid]
        total = matrix.sum()
        return {"pixel_accuracy": (intersection.sum() / total).item() if total else 0.0,
                "mean_iou": iou[valid].mean().item() if valid.any() else 0.0,
                "class_iou": iou.tolist()}
