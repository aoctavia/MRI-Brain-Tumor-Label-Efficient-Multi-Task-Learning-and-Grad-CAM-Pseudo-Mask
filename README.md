# From No Labels to Insight: Self-Supervised Brain Tumor MRI Analysis

> **Can AI learn to detect and segment brain tumors without expensive annotations?**  
> This project is my answer — combining **self-supervised learning**, **Grad-CAM pseudo-masks**, and **multi-task training** to build a label-efficient medical imaging pipeline.

---

## 🎯 Why I Built This
During my Master's in Medical Physics, I saw firsthand how **manual labeling of medical images** is:
- Expensive 💰
- Time-consuming ⏳
- Often inconsistent across experts 🩺

This project aims to **minimize the need for manual labels** by using self-supervision and automated pseudo-label generation.  
By reducing annotation costs, we can make AI in healthcare more **accessible, faster to develop, and easier to deploy** in real clinical settings.

---

## 🧠 Project Overview
Pipeline steps:
1. **Self-Supervised Pretraining (SimCLR)** — trains a ResNet18 encoder without labels.
2. **Grad-CAM Pseudo-Mask Generation** — creates approximate segmentation masks without manual annotation.
3. **Multi-Task Learning** — jointly trains:
   - A classification head (4 tumor classes)
   - A UNet decoder for segmentation
4. **Evaluation, Robustness, and Calibration**
5. **Deployment Ready** — export to TorchScript for lightweight inference.

---

## 📊 Key Results (Test Set)
- **Accuracy**: **89.09%**
- **Macro F1**: **88.37%**
- **ROC-AUC (macro)**: **0.9842**  
- **Segmentation Dice (vs pseudo-mask)**: **0.533**

**Per-Class Performance:**
| Class           | Precision | Recall | F1-score |
|-----------------|-----------|--------|----------|
| Glioma          | 0.9255    | 0.8700 | 0.8969   |
| Meningioma      | 0.8715    | 0.7092 | 0.7820   |
| No Tumor        | 0.9076    | 0.9704 | 0.9379   |
| Pituitary Tumor | 0.8559    | 0.9900 | 0.9181   |

**Baseline vs Multi-Task:**
| Model | Accuracy | Macro F1 |
|---|---:|---:|
| **ResNet18 Supervised (single-task)** | 0.8172 | 0.8021 |
| **SSL + Pseudo-mask + Multi-task**    | **0.8528** | **0.8424** |

> Multi-task learning improves recall for *meningioma* and overall macro F1.

---

## 🖼️ Visual Examples
| Original MRI | Grad-CAM Overlay | Pseudo-Mask | Segmentation Output |
|--------------|-----------------|-------------|----------------------|
| ![Original](figures_samples/original_glioma.jpg) | ![GradCAM](figures_samples/gradcam_glioma.jpg) | ![PseudoMask](figures_samples/mask_glioma.jpg) | ![Output](figures_samples/output_glioma.jpg) |

---

## 🚀 Quickstart
```bash
pip install -r requirements.txt
````

**Run inference on a single MRI:**

```bash
python infer_one.py \
  --model checkpoints/multitask_resnet18.pt \
  --image data_examples/sample.png \
  --out figures_samples/pred_overlay.png
```

**Export TorchScript:**

```bash
python export_torchscript.py \
  --model checkpoints/multitask_resnet18.pt \
  --out checkpoints/model_scripted.pt
```

---

## 🧠 Novel Contributions

* **Label-efficient approach** using self-supervision
* **Segmentation without GT masks** via Grad-CAM pseudo-labels
* **Joint learning** to boost classification recall
* **Robustness testing** under noise & motion artefacts
* **Confidence calibration** for safer predictions

---

## 🌍 Impact & Future Work

**Impact:**

* Lowering the barrier for AI adoption in hospitals with limited labeled data
* Enabling AI models that are both **accurate** and **interpretable** for clinicians
* Providing a reusable pipeline for other medical imaging domains (CT scans, ultrasound)

**Future Work:**

* Incorporate **semi-supervised learning** with small labeled subsets
* Extend to **3D MRI volumetric data**
* Integrate **clinical metadata** (age, symptoms) for multi-modal learning
* Deploy a lightweight web demo for radiologists to test the model

---

## 📬 Contact

* **Author**: Aulia Octavia
* **Background**: Physics, AI/ML for medical
* **Email**: [auliaoctavvia@gmail.com](mailto:auliaoctavvia@gmail.com)
* **Portfolio**: [LinkedIn](#) | [GitHub](#)

> *"Turning weak supervision into strong medical insights — one MRI at a time."*
