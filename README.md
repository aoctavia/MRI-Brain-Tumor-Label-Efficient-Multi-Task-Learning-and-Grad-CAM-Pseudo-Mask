# MRI Brain Tumor – SSL + Pseudo-Mask + Multi-Task (Classification + Segmentation)

**Short portfolio project** for PhD applications in Medical Physics / AI-ML.

- **Pipeline**: Self-Supervised Learning (SSL) → Grad-CAM **pseudo-masks** → **Multi-task** training → Evaluation → Robustness/Calibration → Export to TorchScript
- **Dataset (example structure on Drive)**:
  ```
  Training/
      meningioma/ glioma/ pituitary/ notumor/
  Testing/
      meningioma/ glioma/ pituitary/ notumor/
  ```

## 🔧 Results (Test Set)
- **Accuracy**: **0.8528**
- **Macro F1**: **0.8424**
- **Per-Class**:
  - Glioma: P=0.9500, R=0.7600, F1=0.8444 (n=300)
  - Meningioma: P=0.8354, R=0.6634, F1=0.7395 (n=306)
  - No Tumor: P=0.9151, R=0.9580, F1=0.9361 (n=405)
  - Pituitary Tumor: P=0.7401, R=0.9967, F1=0.8494 (n=300)

**Confusion Matrix** (as reported):

|                | Pred Glioma | Pred Meningioma | Pred No Tumor | Pred Pituitary |
|----------------|-------------|------------------|---------------|----------------|
| **True Glioma**     | 228         | 45               | 0             | 27             |
| **True Meningioma** | 67          | 203              | 1             | 35             |
| **True No Tumor**   | 0           | 11               | 388           | 6              |
| **True Pituitary**  | 0           | 1                | 0             | 299            |

> Additional metrics: **ROC-AUC (macro)** ≈ 0.9842; **Segmentation Dice vs pseudo-mask**: 0.533.

## 🆚 Baseline vs Multi-task
| Model | Accuracy | Macro F1 |
|---|---:|---:|
| **ResNet18 Supervised (single-task)** | 0.8172 | 0.8021 |
| **SSL + Pseudo-mask + Multi-task** | **0.8528** | **0.8424** |

> Multi-task improves recall for *meningioma* and overall macro F1.

## 📁 Repository Structure
```
.
├─ notebooks/
│  └─ 03_evaluate_and_visualize.ipynb
├─ scripts/
│  ├─ infer_one.py
│  └─ export_torchscript.py
├─ figures_samples/        # saved overlays go here
├─ checkpoints/            # place your *.pt / *.pth here
├─ pseudo_masks/           # grad-cam pseudo masks (optional)
├─ data_examples/          # a few PNGs for quick demo
├─ requirements.txt
└─ README.md
```

## 🚀 Quickstart (Colab or local)
```bash
pip install -r requirements.txt
```

### Inference (single image)
```bash
python scripts/infer_one.py   --model checkpoints/multitask_resnet18.pt   --image data_examples/sample.png   --out figures_samples/pred_overlay.png
```

### Export TorchScript
```bash
python scripts/export_torchscript.py   --model checkpoints/multitask_resnet18.pt   --out checkpoints/model_scripted.pt
```

### Evaluate & Visualize
Open the notebook:
```
notebooks/03_evaluate_and_visualize.ipynb
```
It will:
- load predictions & targets (CSV or generated on-the-fly),
- compute classification report + confusion matrix,
- save 3–5 visual examples per class with Grad-CAM overlay (if available).

## 🧠 Novelty Highlight
- **SSL encoder**: label-efficient pretraining for medical images.
- **Grad-CAM pseudo-masks**: segmentation without GT masks.
- **Joint multi-task**: improved recognition via spatial inductive signals.
- **Robustness & calibration**: tested with noise/blur + ECE.

## 📨 Contact & CV
- **Portfolio owner**: Aulia Octavia — Medical AI / Mobile Dev background
- **Email**: auliaoctavvia@gmail.com
- **CV line suggestion**:
  > *Developed an end-to-end brain MRI pipeline (SSL → pseudo-mask → multi-task), achieving 85.3% test accuracy and 0.842 macro F1; delivered TorchScript export, inference script, and visual interpretability.*

