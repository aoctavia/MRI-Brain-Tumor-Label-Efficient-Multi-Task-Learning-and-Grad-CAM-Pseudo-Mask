# 🧠 MRI Brain Tumor – Self-Supervised + Pseudo-Mask + Multi-Task Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/username/repo-name)

> **Short portfolio project** for PhD applications in **Medical Physics / AI-ML**, demonstrating a label-efficient approach for **classification and segmentation** of brain MRI tumors.

---

## Background
Brain tumor diagnosis from MRI is challenging due to:
- Limited **labeled data**, especially segmentation masks.
- The need for **robust** and **interpretable** models in clinical settings.

**This project addresses the gap by:**
1. Using **Self-Supervised Learning (SSL)** for feature pretraining without labels.
2. Generating **pseudo-segmentation masks** via Grad-CAM.
3. Training a **joint multi-task network** for classification + segmentation.

---

## Dataset
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor  
- **Train:** 4,855 | **Validation:** 857 | **Test:** 1,311  
- **Structure:**
```

Training/
meningioma/ glioma/ pituitary/ notumor/
Testing/
meningioma/ glioma/ pituitary/ notumor/

````

---

## Results (Test Set)
- **Accuracy:** 0.8909
- **Macro F1:** 0.8837
- **ROC-AUC (macro):** 0.9842
- **Segmentation Dice (vs pseudo-mask):** 0.533

| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| Glioma          | 0.9255    | 0.8700 | 0.8969   | 300     |
| Meningioma      | 0.8715    | 0.7092 | 0.7820   | 306     |
| No Tumor        | 0.9076    | 0.9704 | 0.9379   | 405     |
| Pituitary Tumor | 0.8559    | 0.9900 | 0.9181   | 300     |

**Confusion Matrix:**

|                | Pred Glioma | Pred Meningioma | Pred No Tumor | Pred Pituitary |
|----------------|-------------|-----------------|---------------|----------------|
| **True Glioma**     | 261         | 27              | 0             | 12             |
| **True Meningioma** | 15          | 217             | 40            | 34             |
| **True No Tumor**   | 4           | 4               | 393           | 4              |
| **True Pituitary**  | 2           | 1               | 0             | 297            |

---

## Baseline vs Multi-task
| Model                              | Accuracy | Macro F1 |
|------------------------------------|----------|----------|
| ResNet18 Supervised (single-task)  | 0.8172   | 0.8021   |
| **SSL + Pseudo-mask + Multi-task** | **0.8528** | **0.8424** |

> Multi-task learning improved recall for *meningioma* and boosted macro F1 by ~4%.

---

## Visual Highlights
**Sample Grad-CAM Overlays**
<p align="center">
<img src="figures_samples/glioma_sample.png" width="200"/>
<img src="figures_samples/meningioma_sample.png" width="200"/>
<img src="figures_samples/pituitary_sample.png" width="200"/>
<img src="figures_samples/no_tumor_sample.png" width="200"/>
</p>

**t-SNE Embedding**
<p align="center">
<img src="figures_samples/tsne_plot.png" width="400"/>
</p>

---

## Novelty Highlights
- **SSL encoder**: Pretrained without labels, boosting feature quality.
- **Grad-CAM pseudo-masks**: Enable segmentation training without GT masks.
- **Joint multi-task**: Improves classification via spatial feature learning.
- **Robustness & calibration**: Evaluated with noise, blur, and Expected Calibration Error (ECE).

---

## Quickstart (Colab or Local)

### 1. Install Requirements
```bash
pip install -r requirements.txt
````

### 2. Inference (Single Image)

```bash
python scripts/infer_one.py \
    --model checkpoints/multitask_resnet18.pt \
    --image data_examples/sample.png \
    --out figures_samples/pred_overlay.png
```

### 3. Export to TorchScript

```bash
python scripts/export_torchscript.py \
    --model checkpoints/multitask_resnet18.pt \
    --out checkpoints/model_scripted.pt
```

### 4. Evaluate & Visualize

Open the notebook:

```
notebooks/03_evaluate_and_visualize.ipynb
```

It will:

* Load predictions & targets (CSV or generated on-the-fly).
* Compute classification report + confusion matrix.
* Save 3–5 visual examples per class with Grad-CAM overlay (if available).

---

## Repository Structure

```
.
├─ notebooks/
│  └─ 03_evaluate_and_visualize.ipynb
├─ scripts/
│  ├─ infer_one.py
│  └─ export_torchscript.py
├─ figures_samples/        # Saved overlays
├─ checkpoints/            # *.pt / *.pth models
├─ pseudo_masks/           # Grad-CAM pseudo masks
├─ data_examples/          # Example MRI slices
├─ requirements.txt
└─ README.md
```

---

## Contact & CV

**Portfolio Owner:** Aulia Octavia — Medical AI / Mobile Dev background
**Email:** [auliaoctavvia@gmail.com](mailto:auliaoctavvia@gmail.com)
