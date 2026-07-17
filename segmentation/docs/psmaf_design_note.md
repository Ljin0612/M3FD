# PSMAF-Net Design Note

PSMAF-Net is the tentative name of the proposed RGB-IR fusion model.

Full name: **Pseudo-Semantic guided Multi-scale Adaptive Fusion Network**.

Chinese name: **伪语义引导的多尺度自适应跨模态融合网络**.

Current positioning:
- Pseudo-semantic guidance belongs to the upstream fusion model, not to a detection head or segmentation head.
- Multi-scale adaptation belongs to the upstream fusion model, not to a detection-specific neck.
- Detection and segmentation are downstream tasks for validating the shared RGB-IR fusion representation.

This segmentation branch currently only establishes the segmentation task structure and baselines. The final PSMAF-Net modules will be implemented later.
