# Object Recognition: Traditional CV vs Deep Learning

A comparative study of **classical computer vision** (SIFT + BoVW + SVM) versus **deep learning** (custom CNN and ResNet-18 transfer learning) for object recognition, evaluated on two datasets with contrasting characteristics: CIFAR-10 and the RGB-D Object Dataset.

---

## Overview

This project investigates the trade-offs between traditional and modern approaches to object recognition across two key dimensions: a controlled benchmark setting (CIFAR-10) and a real-world, high-variability setting (RGB-D). The comparison covers accuracy, generalization, class-wise performance, and practical considerations such as data requirements and computational cost.

| Aspect | Traditional (SIFT + BoVW + SVM) | Deep Learning (CNN / ResNet-18) |
|--------|----------------------------------|----------------------------------|
| Feature extraction | Handcrafted (SIFT keypoints) | Learned end-to-end |
| Interpretability | High | Low |
| Data requirement | Low | High |
| Accuracy (CIFAR-10) | 28.71% | 78.84% |
| Accuracy (RGB-D) | 34.95% | 72.36% |
| Compute cost | Low | High |

---

## Datasets

### CIFAR-10
60,000 RGB images (32×32) across 10 categories — animals and vehicles. Pre-split into 50,000 training and 10,000 test images. 20% of training was held out for validation. Images were upscaled to 224×224 for SIFT keypoint coverage and CNN input compatibility.

### RGB-D Object Dataset (Cropped Evaluation Set)
51 object categories with multiple instances per category, each captured across three video sequences (subsampled every 5th frame). Images are tightly cropped around the object.

**Data leakage prevention:** Since multiple frames per instance are highly correlated, random per-image splitting was avoided. Instead, **instance-based splitting** was used — ensuring all frames of a given object instance (e.g., `apple_1_*.png`) appear exclusively in either the training or test set, not both.

---

## Methodology

### Traditional Pipeline: SIFT + BoVW + SVM

**1. Feature Extraction (SIFT)**
SIFT (Scale-Invariant Feature Transform) was used to extract local descriptors from grayscale images. It was chosen for robustness to scale, rotation, and illumination changes — relevant for both datasets.

**2. Bag-of-Visual-Words (BoVW)**
SIFT descriptors across all training images were aggregated and clustered using **MiniBatch K-Means** with a vocabulary of 100 visual words (chosen as a trade-off between representational power and training time — larger vocabularies risk overfitting and dimension explosion). Each image is then encoded as an L2-normalized histogram over this vocabulary.

**3. SVM Classification**
BoVW histograms were standardized with `StandardScaler` and fed into a **linear SVM**. A linear kernel was preferred over RBF as preliminary tests showed minimal accuracy gains at significantly higher cost. No formal grid search was conducted due to computational constraints.

### Deep Learning Pipeline

**CIFAR-10 — Custom CNN from Scratch**
Given CIFAR-10's 50,000 training images, a small CNN was trained from scratch:
- 3 convolutional layers with ReLU activations and MaxPooling
- Flattening layer → 2 fully connected layers → Softmax (10 classes)
- Optimizer: Adam (lr = 1e-4), batch size = 32, ~10–20 epochs

**RGB-D — Transfer Learning with ResNet-18**
Given the smaller RGB-D evaluation set, a pretrained ResNet-18 (ImageNet weights) was fine-tuned:
- Final FC layer replaced to output 51 classes
- Base layers optionally frozen or fine-tuned
- Learning rate scheduling applied; lr = 1e-4 to avoid overwriting pretrained weights

---

## Results

### CIFAR-10

| Method | Test Accuracy | Macro F1 |
|--------|--------------|----------|
| SIFT + BoVW + SVM | 28.71% | 0.29 |
| Custom CNN | **78.84%** | **0.79** |

**CNN per-class highlights:**
- Best: Ship (0.92), Automobile (0.90)
- Weakest: Cat (0.60), Bird (0.70)

The traditional pipeline struggled primarily due to CIFAR-10's low native resolution (32×32), which limits SIFT keypoint coverage, and the spatial context loss inherent to BoVW. Widespread misclassification occurred between visually similar animal categories (cat/dog/horse).

### RGB-D Object Dataset

| Method | Test Accuracy | Macro F1 | Weighted F1 |
|--------|--------------|----------|-------------|
| SIFT + BoVW + SVM | 34.95% | 0.30 | 0.34 |
| ResNet-18 (Transfer Learning) | **72.36%** | **0.68** | **0.72** |

**ResNet-18 highlights:**
- 31,696 RGB frames evaluated across 51 classes
- Most classes achieved precision/recall > 0.90 (e.g., `food_cup`, `toothpaste`, `sponge`)
- Weakest classes: `marker`, `mushroom`, `comb` (F1 < 0.25) — likely due to class imbalance or visual similarity

**Traditional method highlights:**
- Some distinctive classes performed well: `marker` (0.56), `pliers` (0.57), `glue_stick` (0.58)
- Many classes scored F1 ≈ 0.10–0.20; some (e.g., `mushroom`, `peach`, `pitcher`) received near-zero correct predictions
- Limitations: BoVW ignores spatial layout and is sensitive to pose, scale, and background variation across RGB-D frames

---

## State of the Art Context

Deep learning now dominates robotic vision. Key trends include:

- **Object detection:** YOLOv7 (real-time), Faster R-CNN (two-stage pipeline)
- **Segmentation:** Mask R-CNN, DeepLab for pixel-wise classification in grasping and navigation
- **3D recognition:** PointNet / PointNet++ operating directly on LiDAR and RGB-D point clouds
- **Transformers:** DETR, Swin Transformer for long-range visual dependencies

**Remaining challenges:** high data requirements, limited interpretability, and hardware constraints for edge deployment. Emerging directions include hybrid handcrafted+CNN models, self-supervised learning (SimCLR, BYOL), and Edge AI via model pruning and quantization.

---

## Code

All experiments are available as Kaggle notebooks:

| Experiment | Link |
|------------|------|
| CIFAR-10 — SIFT + BoVW + SVM | [kaggle.com/code/mananmal/cifar-10-traditional](https://www.kaggle.com/code/mananmal/cifar-10-traditional) |
| CIFAR-10 — Custom CNN | [kaggle.com/code/mananmal/cifar-10-cnn](https://www.kaggle.com/code/mananmal/cifar-10-cnn) |
| RGB-D — SIFT + BoVW + SVM | [kaggle.com/code/mananmal/rgb-d-traditional](https://www.kaggle.com/code/mananmal/rgb-d-traditional) |
| RGB-D — ResNet-18 | [kaggle.com/code/mananmal/cnn-for-rgb-d](https://www.kaggle.com/code/mananmal/cnn-for-rgb-d) |

---

## References

- Ren, S. et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*.
- Wang, C.-Y. et al. (2023). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art. *arXiv*.
- Qi, C.R. et al. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *CVPR*.
- Carion, N. et al. (2020). End-to-End Object Detection with Transformers. *ECCV*.
- Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*.

---

## Author

**Manan Mal**  
Feel free to open an issue or reach out with questions or suggestions.
