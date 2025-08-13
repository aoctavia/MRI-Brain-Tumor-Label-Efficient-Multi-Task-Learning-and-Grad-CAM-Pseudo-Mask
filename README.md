# MRI Brain Tumor – Label-Efficient Multi-Task Learning with Self-Supervision & Grad-CAM Pseudo-Masks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/<username>/<repo>/blob/main/Self_Supervised_Pretraining_+_Multi_Task_Segmentation_Classification.ipynb)

**Proof-of-concept** for a label-efficient brain tumor classification and segmentation pipeline, designed for **AI in Medical Imaging** research.  
This project demonstrates how self-supervised learning, Grad-CAM pseudo-masks, and multi-task training can be combined to reduce annotation cost and improve model interpretability.

---

## 🌍 Why This Matters
In medical imaging, obtaining large-scale **pixel-level annotations** (e.g., tumor masks) is costly, time-consuming, and requires expert radiologists.  
This work shows how we can:
- Use **self-supervised pretraining** to leverage unlabeled data.
- Generate **pseudo-masks** via Grad-CAM for weakly supervised segmentation.
- Train a **multi-task model** for classification + segmentation **without manual mask annotations**.

Potential applications:
- Rapid prototyping for **MRI analysis research**.
- Reducing reliance on expensive labeled datasets.
- A foundation for further research on **domain adaptation**, **federated learning**, or **multi-modal MRI**.

---

## 📊 Results (Test Set)

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| Glioma           | 0.9255    | 0.8700 | 0.8969   | 300     |
| Meningioma       | 0.8715    | 0.7092 | 0.7820   | 306     |
| No Tumor         | 0.9076    | 0.9704 | 0.9379   | 405     |
| Pituitary Tumor  | 0.8559    | 0.9900 | 0.9181   | 300     |
| **Overall Acc.** | **0.8909**|        |          | 1311    |

**Macro F1**: **0.8837**  
**ROC-AUC (macro)**: **0.9842**  
**Segmentation Dice vs pseudo-mask**: 0.533

---

## 🆚 Baseline vs Multi-task

| Model                                      | Accuracy | Macro F1 |
|--------------------------------------------|----------|----------|
| ResNet18 Supervised (single-task)          | 0.8172   | 0.8021   |
| **SSL + Pseudo-mask + Multi-task**         | **0.8909**| **0.8837** |

---

## 🖼 Visual Examples

| Original MRI | Grad-CAM Pseudo-Mask | Predicted Segmentation |
|--------------|----------------------|------------------------|
| ![orig](figures_samples/sample_orig.jpg) | ![cam](figures_samples/sample_cam.jpg) | ![seg](figures_samples/sample_seg.jpg) |

---

## 📁 Repository Structure
```

.
├─ Self\_Supervised\_Pretraining\_+\_Multi\_Task\_Segmentation\_Classification.ipynb
├─ export\_torchscript.py
├─ infer\_one.py
├─ checkpoints/            # trained models (\*.pt)
├─ figures\_samples/        # visualization results
├─ pseudo\_masks/           # Grad-CAM pseudo masks
├─ requirements.txt
└─ README.md

````

---

## 🚀 Quickstart

**Install dependencies**
```bash
pip install -r requirements.txt
````

**Run notebook (Colab or local)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo>/blob/main/Self_Supervised_Pretraining_+_Multi_Task_Segmentation_Classification.ipynb)

**Single-image inference**

```bash
python infer_one.py \
  --model checkpoints/multitask_resnet18.pt \
  --image data_examples/sample.png \
  --out figures_samples/pred_overlay.png
```

**Export TorchScript**

```bash
python export_torchscript.py \
  --model checkpoints/multitask_resnet18.pt \
  --out checkpoints/model_scripted.pt
```

---

## 🧠 Novelty Highlight

* **SSL encoder**: Uses SimCLR-based pretraining to reduce label requirements.
* **Grad-CAM pseudo-masks**: Weakly-supervised segmentation without ground truth masks.
* **Joint multi-task learning**: Improves classification by leveraging spatial context.
* **Robustness testing**: Evaluated with noise & motion blur.
* **Model calibration**: Low ECE (0.0253) on test set.

---

## 👩‍💻 Author

**Aulia Octavia** – Background in Medical AI & Mobile Development
📧 Email: [auliaoctavvia@gmail.com](mailto:auliaoctavvia@gmail.com)
💼 [LinkedIn](https://linkedin.com/in/aoctavia) | [Portfolio GitHub](https://github.com/aoctavia)

> *Developed an end-to-end brain MRI pipeline (SSL → pseudo-mask → multi-task), achieving 89% test accuracy and 0.883 macro F1; delivered TorchScript export, inference script, and visual interpretability.*

---
