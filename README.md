# Multilabel Movie Genre Classification from Posters

This repository contains the code and dataset splits for our paper: **"Multilabel Movie Genre Classification from Poster Images Using Deep-Learning Frameworks."**

In this project, we adapt and extend the methodology proposed by Unal et al. (2023) to a constrained dataset of 300 IMDb movie posters, predicting across 14 simultaneous genre labels.

## Key Contributions
* **Simplified Linear Probe:** To combat the data scarcity (250 training images) and prevent severe overfitting, we replaced the multi-layer perceptron head used in prior literature with a regularized linear probe (`BatchNorm1d → Dropout(p=0.5) → Linear`).
* **Transfer Learning Analysis:** We evaluated 5 state-of-the-art CNN backbones (ResNet50, VGG16, ConvNeXt, DenseNet121, MobileNet).
* **Catastrophic Forgetting Proof:** Our experiments empirically demonstrate that full fine-tuning on extremely limited data destroys pretrained ImageNet features, whereas frozen-backbone transfer learning yields robust generalization.

## Final Results (Test Set)
We evaluated the models using **Hamming Loss (HL)** as the primary metric (lower is better) and a decision threshold of $\tau = 0.5$.

| Model | Phase | HL (τ=0.5) ↓ | HL (Top-4) ↓ | F1-Score ↑ | Accuracy ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | **Frozen** | **0.1929** | **0.1757** | **0.6423** | 0.0400 |
| **VGG16** | Frozen | 0.2000 | 0.2043 | 0.6084 | **0.1200** |
| **ConvNeXt** | Frozen | 0.2014 | 0.1900 | 0.6369 | 0.0200 |

* **ResNet50** achieved the best overall balance between Precision and Recall, while **VGG16** adopted a more conservative strategy, achieving the highest exact-match Accuracy.*

## Authors

* Devan Westley (Universitas Gadjah Mada)

* Juliette Faurie (Universitas Gadjah Mada)

* Sei Louis Bayle (Universitas Gadjah Mada)
