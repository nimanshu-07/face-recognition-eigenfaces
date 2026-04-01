# face-recognition-eigenfaces
Face recognition from scratch using PCA, LDA, Kernel PCA, and Kernel LDA on the Yale Face Database, with visualizations and kNN-based evaluation.



# Face Recognition with PCA, LDA, Kernel PCA, and Kernel LDA

A classical machine learning face recognition project implemented with NumPy, OpenCV, and kNN on the Yale Face Database. This project compares linear and kernel-based dimensionality reduction methods for recognition, reconstruction, and feature-space visualization.

## Overview

This project implements and evaluates:

- **PCA (Eigenfaces)**
- **LDA (Fisherfaces)**
- **Kernel PCA**
- **Kernel LDA** using **Kernel PCA → LDA**
- **kNN classification** in projected feature space

The goal is to study how different projection methods affect face reconstruction, class separability, and recognition accuracy.

---

## Dataset

The project uses the **Yale Face Database**.

- **Total images:** 165
- **Subjects:** 15
- **Training set:** 135 images
- **Testing set:** 30 images

### Preprocessing
Each image is:

- resized to **64 × 64**
- converted to **grayscale**
- normalized to **[0, 1]**
- flattened into a vector of size **4096**

---

## Methods

### 1. PCA (Eigenfaces)
PCA finds directions of maximum variance in the image space and projects face images onto a lower-dimensional basis.

Used for:
- dimensionality reduction
- image reconstruction
- feature extraction for classification

### 2. LDA (Fisherfaces)
LDA finds directions that maximize class separability using label information.

Used for:
- supervised feature extraction
- improving identity separation over PCA

### 3. Kernel PCA
Kernel PCA applies PCA in an implicit nonlinear feature space using an RBF kernel.

Used for:
- nonlinear dimensionality reduction
- comparing linear vs nonlinear unsupervised methods

### 4. Kernel LDA
Kernel LDA is implemented through the practical pipeline:

**Kernel PCA → LDA → kNN**

Used for:
- nonlinear supervised discrimination
- improving cluster separation in feature space

### 5. kNN Classification
After projection, classification is performed using **k-nearest neighbors (kNN)** with Euclidean distance.

---

## Project Structure

```text
face-recognition-eigenfaces/
├── assets/
│   ├── eigenfaces.png
│   ├── pca_reconstruction.png
│   ├── fisherfaces.png
│   ├── lda_reconstruction.png
│   ├── pca_feature_space.png
│   ├── lda_feature_space.png
│   └── kernel_lda_feature_space.png
├── notebooks/
│   └── face_recognition.ipynb
├── src/
│   └── face_recognition.py
├── report/
│   └── report.pdf
├── requirements.txt
```
---


## Results
| Method | Best Setting | Accuracy |
|---|---|---:|
| PCA + kNN | `m = 50, k = 5` | 0.900 |
| LDA + kNN | `mlda = 14, k ∈ {1, 3, 5}` | 0.967 |
| Kernel PCA + kNN (RBF) | `m = 20/30, k = 5` | 0.867 |
| Kernel LDA + kNN (RBF) | `kpca dim = 50, k = 5` | 1.000 |

---

## Key Observations
1. PCA captures dominant variance but does not explicitly separate identities.
2. LDA improves recognition because it uses label information and maximizes between-class separation.
3. Kernel PCA introduces nonlinear features, but as an unsupervised method it does not necessarily improve identity discrimination.
4. Kernel LDA achieves the best performance by combining nonlinear mapping with supervised discrimination.

---


## Visualizations

