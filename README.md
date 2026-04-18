# Uncertainty-Aware Skin Cancer Classification

### Two-Stage CNN + MC Dropout + AUQantO + Explainable AI

## Project Presentation Videos

Phase 1 (Mid Evaluation): [phase-1-video ](https://drive.google.com/file/d/1jgqHtZ9AEP6PGaVJHEq0rjwx3Waa9lgw/view)

Phase 2 (Final Evaluation): [phase-2-video ](https://drive.google.com/file/d/1uynUfQW45blza4G_3O9S30IeliyfOzky/view)

---
## Overview

This project implements an **uncertainty-aware deep learning framework** for **skin cancer classification (Benign vs Malignant)**.

Instead of blindly predicting, the model:

* **Predicts**
* **Estimates uncertainty**
* **Rejects unreliable predictions**
* **Explains its decisions**

The approach is inspired by the **AUQantO framework**  and extends it with:

* Two-stage CNN architecture
* Monte Carlo Dropout
* COBYLA optimization
* Explainable AI (Integrated Gradients + Spatial Uncertainty)

---

## Architecture

### Stage 1: Patch-Level CNN

* Image → split into patches (e.g., 3×3 grid)
* Each patch → passed through CNN
* Extracts **local features**

### Stage 2: Image-Level Classifier

* Combine patch features using:

  * Mean pooling
  * Max pooling
* Fully connected layers → final prediction

---

## Key Components

### Uncertainty Estimation (MC Dropout)

* Dropout active during inference
* Multiple forward passes (e.g., 50 runs)
* Outputs:

  * Mean prediction
  * Uncertainty (entropy)

---

### AUQantO (Selective Prediction)

* Introduces threshold **λ (lambda)**
* Splits predictions:

| Condition | Action    |
| --------- | --------- |
| unc < λ   |  Accept  |
| unc ≥ λ   |  Reject |

* Threshold optimized using **COBYLA**

---

### Explainable AI

#### Integrated Gradients

* Highlights **important pixels**
* Shows *where model is looking*

#### Spatial Uncertainty Maps

* Shows **uncertain regions**
* Patch-wise variance visualization

---

## Results

| Metric            | Value      |
| ----------------- | ---------- |
| Baseline Accuracy | **80.3%**  |
| Accepted Accuracy | **82.45%** |
| Threshold (λ)     | **0.685**  |
| Coverage          | **91.52%** |
| Rejection Rate    | **8.48%**  |

**+2.15% improvement** after rejecting uncertain samples

---

## Repository Structure

```
.
├── auqanto.ipynb          # Main implementation notebook
├── ip_finalReport.pdf     # Final project report
├── paper.pdf              # AUQantO research paper
├── README.md              # Project documentation
```

---

## Implementation Details

* Framework: **PyTorch**
* Image size: `224 × 224`
* Patch size: `112 × 112`
* MC Runs: `50`
* Optimizer: `Adam`
* Learning rate: `1e-4`

---

## Pipeline

```
Image → Patches → CNN → Features
       ↓
Aggregation (Mean + Max)
       ↓
Prediction
       ↓
MC Dropout (multiple runs)
       ↓
Uncertainty (Entropy)
       ↓
Threshold (λ)
       ↓
Accepted / Rejected
       ↓
Explainability (IG + Heatmaps)
```

---

## Sample Behavior

### Accepted Samples

* Focused IG heatmap
* Low uncertainty
* Reliable prediction

### Rejected Samples

* Noisy/ scattered IG
* High uncertainty
* Model abstains (good behavior)

---

## 🚀 Why This Project Matters

Traditional models:

* Always predict
* No confidence

Our model:

* Knows when it's unsure
* Avoids wrong predictions
* Provides explanations

Crucial for **medical AI systems**

---

## Limitations

* Higher inference time (MC Dropout)
* Threshold depends on data distribution

---


## References

* AUQantO Paper: https://www.open-access.bcu.ac.uk/14826/1/1-s2.0-S1568494623006841-main.pdf 

---

## Authors

* Rohini (202311019)
* Jasmeen Kaur (202311037)
* Hasini Reddy (202311040)

**Under the guidance of:**
Dr. Jignesh Patel
IIIT Vadodara 

---

## Final Note

This project focuses not just on **accuracy**, but on **trustworthy AI** —
a critical step toward real-world deployment in healthcare.

---
