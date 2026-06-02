# IDEA4
<div align="center">

# 🔩 SteelVision AI — Industrial Surface Defect Detection

### *A complete Applied AI research pipeline that compares four machine learning paradigms — MLP, Ensemble Learning, CNN, and Unsupervised Clustering — for automated steel surface defect classification, achieving 95.28% test accuracy on the NEU Surface Defect Database.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4c72b0?style=flat-square)](https://seaborn.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

## GitHub Preview Notice

This repository contains large Jupyter Notebook files. GitHub may not render the notebook directly due to file size.

If the preview is unavailable: clone the repo locally and open `N1364759_AAI_Shah.ipynb` in Jupyter or VS Code. The written report `N1364759_Pratham_Shah_AAI.pdf` contains all results, tables, figures, and analysis in full.

---

## Executive Summary

This project implements and systematically compares four distinct AI paradigms for classifying steel surface defects using the **NEU Surface Defect Database** — a benchmark dataset of 1,800 grayscale images across six defect classes. The goal is to determine how feature representation quality and architectural design drive classification performance in an industrial computer vision context.

The pipeline progresses from a constrained Multilayer Perceptron baseline through handcrafted HOG-feature ensemble learning to a fully trained Convolutional Neural Network, demonstrating a clear and measurable accuracy progression: **25.28% → 65.83% → 86.94% → 95.28%**. An unsupervised clustering study completes the comparison, quantifying how much label information contributes to performance.

The central finding — that **feature representation quality matters more than classifier architecture** — has direct implications for industrial deployment decisions. A classical stacking ensemble with strong HOG features outperforms a raw-pixel neural network by over 21 percentage points, yet the CNN's end-to-end learned hierarchy ultimately delivers the highest accuracy trained on only 1,440 images.

**Built with:** Python · TensorFlow/Keras · scikit-learn · OpenCV · scikit-image · SciPy · Seaborn · Matplotlib
**Dataset:** 1,800 images · 6 classes · 200×200 pixels · perfectly balanced
**Best result:** 95.28% test accuracy · 0.9528 Macro F1-Score

---

## Project Highlights

✅  Processed and explored **1,800 real-world steel surface images** across 6 defect classes with full EDA including KDE plots, cosine similarity heatmaps, and mean-image prototypes

✅  Built and optimised a **constrained MLP** (15→25→25→20→6 neurons) from 25.28% to **65.83%** via a 5-phase sequential hyperparameter search across 17 configurations — with the architecture fixed throughout

✅  Engineered **HOG feature descriptors** (9 orientations, 8×8 cells, 2×2 block L2-normalisation) reducing 40,000-dimensional pixel vectors to ~100–200 PCA components retaining 95% variance

✅  Built a **Stacking Ensemble** (Extra Trees + Random Forest + Gradient Boosting → Logistic Regression meta-learner) achieving **86.94%** accuracy with zero GPU requirement and full interpretability

✅  Designed and trained a **4-block CNN** (32→64→128→256 filters, BatchNorm, progressive dropout 0.20–0.50) with built-in augmentation reaching **95.28% test accuracy** and **0.9528 Macro F1**

✅  Evaluated **3 unsupervised clustering algorithms** (KMeans, Agglomerative, GMM) with Hungarian label matching, ARI, NMI, Silhouette Score, and composite k-selection over k=2–10

✅  Confirmed via cosine similarity analysis (all pairs > 0.97) that **global brightness cannot separate defect classes** — only local spatial texture carries discriminative signal

✅  Produced **20+ publication-quality figures** at 150 DPI: training curves, confusion matrices, per-class F1 charts, t-SNE embeddings, PCA scatter plots, cluster distribution visualisations

✅  Applied **MD5 hash-based duplicate detection** confirming zero data leakage across train/test splits

✅  Delivered a comprehensive **ethical impact analysis** covering employment displacement, domain shift risk, Grad-CAM explainability requirements, and ISO 9001 audit compliance

---

## Business Problem

Steel manufacturing runs at speeds that make manual quality inspection impossible at scale. A typical rolling mill produces kilometres of strip per hour; visual inspection by humans is inconsistent, fatigue-dependent, and becomes a production bottleneck as throughput increases.

The critical distinction this project addresses is not simply "is there a defect?" but "what type of defect is it?" — because different defect types point directly to different root causes in the production process:

| Defect Type | Manufacturing Root Cause | Corrective Action |
|---|---|---|
| Crazing | Thermal stress or quenching rate mismatch | Adjust cooling curve parameters |
| Inclusion | Foreign particle contamination in raw material | Upgrade filtration in smelting |
| Patches | Inconsistent coating or surface treatment | Recalibrate surface treatment line |
| Pitted Surface | Corrosion or mechanical impact during handling | Review transport and storage conditions |
| Rolled-in Scale | Inadequate descaling before rolling pass | Increase descaler pressure/frequency |
| Scratches | Abrasion from handling equipment | Inspect and replace worn guides/rollers |

Multi-class defect classification enables targeted corrective action, predictive maintenance, and closed-loop process control — turning quality inspection from a reactive audit into an intelligent manufacturing feedback system.

**Industry 4.0 relevance:** automotive panels, semiconductor-grade steel, structural beams, and thin-strip electrical steel all require guaranteed surface quality. AI-based inspection systems operating at production-line speeds with greater than 95% accuracy can reduce scrap rates, prevent downstream processing of defective material, and provide digital traceability for regulatory compliance.

---

## Technical Skills Demonstrated

### Deep Learning & Neural Networks
- TensorFlow/Keras — model design, training, evaluation, callbacks
- Convolutional Neural Networks — multi-block architecture, filter hierarchy design
- BatchNormalization, Dropout (progressive 0.20→0.50), EarlyStopping, ReduceLROnPlateau
- Built-in data augmentation — RandomFlip, RandomRotation ±18°
- Adam optimiser with adaptive learning rate scheduling
- Categorical and sparse categorical cross-entropy loss functions

### Classical Machine Learning
- Multilayer Perceptron — prescribed architecture optimisation across 5 phases
- Extra Trees, Random Forest, Gradient Boosting — ensemble base learners
- StackingClassifier with Logistic Regression meta-learner
- RandomizedSearchCV + GridSearchCV — structured hyperparameter tuning
- Stratified K-Fold cross-validation (3-fold and 5-fold)
- L2 regularisation, Dropout, BatchNorm in constrained dense networks

### Computer Vision & Feature Engineering
- OpenCV — image loading, grayscale conversion, resizing (INTER_AREA)
- scikit-image HOG — 9 orientations, 8×8 pixel cells, 2×2 block L2-normalisation
- Principal Component Analysis — 95% variance retention, dimensionality reduction
- StandardScaler and MinMaxScaler — feature normalisation and comparison
- MD5 hash fingerprinting for duplicate detection

### Unsupervised Learning
- KMeans (n_init=15, max_iter=500)
- Agglomerative Clustering — Ward linkage
- Gaussian Mixture Models — diagonal covariance, BIC evaluation
- Hungarian algorithm — optimal label matching for cluster evaluation
- Adjusted Rand Index, Normalised Mutual Information, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score

### Data Analysis & Visualisation
- EDA — class distributions, intensity histograms, KDE plots, cosine similarity heatmaps
- t-SNE embeddings (perplexity=40, PCA-50 pre-reduction for efficiency)
- PCA 2D scatter plots with true vs predicted label overlays
- Training curves, confusion matrices, per-class F1 bar charts
- 20+ publication-quality figures saved at 150 DPI

### Software Engineering
- Reproducibility seed management (SEED=42 across NumPy, TensorFlow, random)
- Modular function design — load_split(), prepare_data(), build_mlp(), run_experiment(), hungarian_accuracy()
- Automated experiment logging with pd.DataFrame result accumulation
- Model checkpointing with .keras format

---

## System Architecture

```
STEELVISION AI — FULL PIPELINE

NEU SURFACE DEFECT DATABASE
1,800 images  6 classes  300/class  200x200px grayscale
Source: Northeastern University (Kaggle)
         |
         v
DATA LOADING & EDA
OpenCV IMREAD_GRAYSCALE  IMG_SIZE=(200,200)  INTER_AREA
Class distribution  Sample grids  Mean images
Pixel histograms  KDE plots  Cosine similarity heatmap
Finding: Cosine similarity > 0.97 across all class pairs
Conclusion: Texture, not brightness, is the discriminative signal
         |
         v
DATA PRE-PROCESSING
Normalise /255 to [0,1]  LabelEncoder  np.newaxis channel dim
Stratified 80/20 split (seed=42)  to_categorical
MD5 duplicate check — 0 duplicates confirmed
Train: 1,440 images    Test: 360 images (60 per class each)
         |
   ______|______________________________________
   |            |                              |
   v            v                              v

TASK A — MLP        TASK B — ENSEMBLE     TASK C — CNN
Flatten: 40k dims   HOG: 9 orient         Input: 96x96x1
                    8x8 cells             Augment: Flip+Rot18
Dense: 15 (ReLU)    2x2 blocks            Block 1: 32 Conv+BN
Dense: 25 (ReLU)    -> StandardScaler     Block 2: 64 Conv+BN
Dense: 25 (ReLU)    -> PCA (95% var)      Block 3: 128 Conv+BN
Dense: 20 (ReLU)                          Block 4: 256 Conv+BN
Dense: 6 (Softmax)  ET + RF + GB          Dense: 256+Drop 0.50
                    -> Logistic Reg.      Softmax: 6
5-phase tuning      RandomizedCV
17 configurations   3-fold StratKF
Adam lr=0.001       
BS=32 EP=50         
EarlySt p=10        

Initial:  25.28%    86.94%                95.28%
Optimised:65.83%    F1: 0.8693            F1: 0.9528
F1: 0.6534
         |
         v
TASK D — CLUSTERING
Full 1,800 images  HOG + StandardScaler + PCA (95% var)
k-selection: Silhouette + Davies-Bouldin + Calinski-Harabasz
Composite score over k=2-10 confirms k=6
KMeans (n_init=15)  Agglomerative (Ward)  GMM (diag covar)
Hungarian algorithm + ARI + NMI + Silhouette evaluation
t-SNE (perplexity=40) + PCA 2D visualisations
Best method (GMM): ~56%   ARI > 0   NMI > 0
```

---

## Dataset Overview

| Property | Value |
|---|---|
| Name | NEU Surface Defect Database |
| Source | Northeastern University via Kaggle |
| Total images | 1,800 |
| Classes | 6 (perfectly balanced — 300 per class) |
| Image format | Grayscale, 200×200 pixels |
| Train split | 1,440 images (80%) — stratified |
| Test split | 360 images (20%) — 60 per class |
| Duplicates | 0 (MD5 hash verified) |
| Missing values | None |
| Class balance | Exact — 16.67% per class |

### Defect Classes

| Class | Physical Description | Manufacturing Implication |
|---|---|---|
| Crazing | Reticulated fine crack network; high-frequency texture | Thermal stress / quenching rate issues |
| Inclusion | Isolated high-contrast dark spots; embedded particles | Raw material contamination |
| Patches | Diffuse low-contrast blotchy regions | Inconsistent surface coating/treatment |
| Pitted Surface | Distributed small cavities; multi-point dark texture | Corrosion or mechanical damage |
| Rolled-in Scale | Elongated dark blobs aligned with rolling direction | Descaling failure before rolling |
| Scratches | Strong directional linear gradients from abrasion | Handling or transport damage |

**Key EDA Finding:** Cosine similarity between all class mean-image prototypes exceeds 0.97, and pixel intensity distributions heavily overlap across all six classes. This proves that global brightness statistics cannot reliably separate defect types — only local spatial texture patterns carry discriminative information. This finding directly justifies the use of HOG features and CNNs over raw-pixel approaches.

---

## Methodology

**Stage 1 — Data Loading and Exploration:** Images loaded with OpenCV IMREAD_GRAYSCALE, resized to 200×200 using INTER_AREA interpolation. EDA produced 8 diagnostic figures: class distribution bar and pie charts, one-per-class sample grids, 5-per-class random galleries, per-class mean images using viridis colourmap, pixel intensity histograms, KDE overlays, cosine similarity heatmaps, and image dimension confirmation plots.

**Stage 2 — Pre-processing:** Six-step pipeline: pixel normalisation ÷255→[0,1]; LabelEncoder integer mapping; channel dimension addition (H×W→H×W×1); stratified 80/20 train/test split with seed=42; one-hot encoding via to_categorical; MD5 hash duplicate detection confirming zero duplicates and no train/test leakage.

**Stage 3 — Task A: MLP Baseline + Optimisation:** Fixed architecture (15→25→25→20→6, ReLU, Softmax) trained with Adam (lr=0.001), batch=32, max 50 epochs, EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5). Initial result: 25.28%. Five-phase sequential optimisation across 17 configurations lifted this to 65.83% — a +40.55pp improvement with the architecture kept unchanged throughout.

**Stage 4 — Task B: Ensemble Learning:** HOG feature extraction (9 orientations, 8×8 cells, 2×2 blocks) on 200×200 images. StandardScaler normalisation followed by PCA retaining 95% variance (~100–200 components). RandomizedSearchCV (n_iter=15, 3-fold stratified CV) over Extra Trees hyperparameter space. Stacking Ensemble — Extra Trees (tuned) + Random Forest + Gradient Boosting (100 estimators, max_depth=4, lr=0.1) with Logistic Regression meta-learner. Result: 86.94% (+21.11pp over MLP).

**Stage 5 — Task C: CNN:** Input resized to 96×96. Built-in augmentation (RandomFlip, RandomRotation ±18°). Four convolutional blocks with filter counts 32→64→128→256, each with BatchNorm + ReLU + MaxPooling + progressive Dropout (0.20→0.25→0.30→0.35). Dense head: 256 units + BatchNorm + ReLU + Dropout 0.50 + 6-class Softmax. Adam (lr=0.001) + ReduceLROnPlateau + EarlyStopping (patience=12, monitor=val_accuracy). Result: 95.28% (+8.34pp over ensemble).

**Stage 6 — Task D: Unsupervised Clustering:** Full 1,800 images. Identical HOG + StandardScaler + PCA pipeline as Task B. Optimal k determined via composite Silhouette + Davies-Bouldin + Calinski-Harabasz score over k=2–10 — k=6 confirmed. Three algorithms evaluated: KMeans (n_init=15), Agglomerative (Ward linkage), GMM (diagonal covariance, reg_covar=1e-4). Hungarian algorithm for label alignment. t-SNE (perplexity=40) and PCA 2D visualisations generated. Best method GMM: ~56%.

---

## Key Results

### Final Performance Comparison

| Task | Method | Paradigm | Test Accuracy | Macro F1 | Improvement |
|---|---|---|---|---|---|
| A (initial) | MLP — prescribed | Supervised | 25.28% | ~0.25 | Baseline |
| A (optimised) | MLP + 5-phase tuning | Supervised | **65.83%** | **0.6534** | +40.55pp |
| B | Stacking Ensemble (HOG+PCA) | Supervised | **86.94%** | **0.8693** | +21.11pp |
| C | CNN (4 conv blocks) | Supervised | **95.28%** | **0.9528** | +8.34pp |
| D | GMM clustering (best) | Unsupervised | ~**56%** | N/A | — |

### CNN Architecture — Task C Best Model

| Block | Configuration | Output Shape | Dropout |
|---|---|---|---|
| Augmentation | RandomFlip + RandomRotation ±18° | 96×96×1 | — |
| Block 1 | Conv→BN→ReLU×2 + MaxPool | 48×48×32 | 0.20 |
| Block 2 | Conv→BN→ReLU×2 + MaxPool | 24×24×64 | 0.25 |
| Block 3 | Conv→BN→ReLU + MaxPool | 12×12×128 | 0.30 |
| Block 4 | Conv→BN→ReLU + MaxPool | 6×6×256 | 0.35 |
| Dense Head | Flatten→Dense(256)→BN→ReLU | 256 | 0.50 |
| Output | Dense(6) + Softmax | 6 | — |

### MLP 5-Phase Optimisation — Task A

| Phase | Variable Tested | Best Finding | Impact |
|---|---|---|---|
| 1 | Image size × scaler | 64×64 + StandardScaler | Critical — fixes 2,667:1 bottleneck |
| 2 | Activation function | ELU / LeakyReLU | Medium — keeps gradients alive |
| 3 | Optimiser × LR | Adam + lr=5e-4 | Medium — reduces oscillation |
| 4 | Regularisation | Dropout 0.2 + BatchNorm | High — stabilises narrow layers |
| 5 | Batch size | Tuned per configuration | Low–medium |

### Feature Representation vs Accuracy

| Feature Representation | Classifier | Test Accuracy |
|---|---|---|
| Raw pixels (200×200 = 40,000 dims) | MLP initial | 25.28% |
| Raw pixels (64×64 = 4,096 dims, optimised) | MLP optimised | 65.83% |
| HOG + StandardScaler + PCA | Stacking Ensemble | 86.94% |
| Learned CNN filters (end-to-end) | CNN | **95.28%** |

The +21pp gain from MLP to Ensemble is entirely driven by switching from raw pixels to HOG descriptors — both models have access to the same 1,440 training images. This isolates the impact of feature representation from the effect of architectural complexity.

---

## Visualisations Produced

| Figure | File | What It Shows |
|---|---|---|
| Class distribution | class_distribution.png | Bar + pie — 300 images per class confirmed |
| Sample per class | sample_per_class.png | One representative image per defect type |
| Image gallery | image_gallery.png | 5 random samples per class (6×5 grid) |
| Mean images | mean_images.png | Per-class average texture prototype (viridis) |
| Intensity histograms | intensity_histograms.png | Pixel distribution per class with mean lines |
| KDE per class | kde_per_class.png | Overlapping intensity distributions all six classes |
| Cosine similarity heatmap | cosine_similarity_heatmap.png | Class mean-image similarity (all > 0.97) |
| Train/test split | train_test_split.png | Stacked bar stratified split confirmation |
| Normalisation check | normalisation_comparison.png | Before/after pixel scaling comparison |
| Pixel statistics | pixel_statistics.png | Per-class mean and std intensity bar charts |
| MLP training curves | mlp_training_curves.png | Accuracy and loss vs epoch — Task A baseline |
| MLP confusion matrix | mlp_confusion_matrix.png | 6×6 predicted vs true — Task A |
| MLP experiment comparison | mlp_experiment_comparison.png | Top 15 experiments ranked by accuracy and F1 |
| Optimised MLP curves | mlp_opt_training_curves.png | Final optimised model training history |
| Optimised MLP confusion matrix | mlp_opt_confusion_matrix.png | Task A optimised — 6×6 heatmap |
| Optimised MLP per-class F1 | mlp_opt_per_class_f1.png | Per-class F1 bars with macro average line |
| Cluster k-selection | taskd_k_selection.png | Elbow, Silhouette, Calinski-Harabasz over k=2–10 |
| Cluster visualisations | taskd_cluster_vis.png | PCA 2D + t-SNE for all 3 methods (2×4 panel) |
| Cluster confusion matrices | taskd_confusion_matrices.png | Hungarian-mapped confusion matrices x3 methods |
| Task A–D comparison | taskd_abcd_comparison.png | Accuracy summary bar chart across all four tasks |

---

## Repository Structure

```
steelvision-ai/
|
|-- N1364759_AAI_Shah.ipynb              # Full implementation — all 4 tasks
|-- N1364759_Pratham_Shah_AAI.pdf        # Written report — all results and analysis
|-- README.md                            # This file
|-- requirements.txt                     # Python dependencies
|
|-- NEU-DET/                             # Dataset directory
|   |-- train/
|   |   `-- images/
|   |       |-- crazing/                 # 240 training images per class
|   |       |-- inclusion/
|   |       |-- patches/
|   |       |-- pitted_surface/
|   |       |-- rolled-in_scale/
|   |       `-- scratches/
|   `-- validation/
|       `-- images/
|           |-- crazing/                 # 60 validation images per class
|           |-- inclusion/
|           |-- patches/
|           |-- pitted_surface/
|           |-- rolled-in_scale/
|           `-- scratches/
|
|-- results/                             # All generated plots (150 DPI PNG)
|   |-- class_distribution.png
|   |-- sample_per_class.png
|   |-- image_gallery.png
|   |-- mean_images.png
|   |-- intensity_histograms.png
|   |-- kde_per_class.png
|   |-- cosine_similarity_heatmap.png
|   |-- train_test_split.png
|   |-- normalisation_comparison.png
|   |-- pixel_statistics.png
|   |-- mlp_training_curves.png
|   |-- mlp_confusion_matrix.png
|   |-- mlp_experiment_comparison.png
|   |-- mlp_opt_training_curves.png
|   |-- mlp_opt_confusion_matrix.png
|   |-- mlp_opt_per_class_f1.png
|   |-- taskd_k_selection.png
|   |-- taskd_cluster_vis.png
|   |-- taskd_confusion_matrices.png
|   |-- taskd_distribution_metrics.png
|   `-- taskd_abcd_comparison.png
|
`-- models/
    `-- mlp_opt_best.keras               # Best optimised MLP checkpoint
```

---

## Installation

**Prerequisites:** Python 3.9+, pip package manager

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/steelvision-ai.git
cd steelvision-ai

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow>=2.12.0
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.7.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.13.0
jupyter>=1.0.0
h5py>=3.8.0
```

**Dataset setup:** Download the NEU Surface Defect Database from Kaggle, extract, and place in the project root as `NEU-DET/` matching the folder structure above.

---

## Usage

```bash
jupyter notebook N1364759_AAI_Shah.ipynb
```

Execute cells sequentially. All plots save automatically. Each section includes markdown documentation explaining purpose, parameters, and expected output.

| Notebook Section | Content |
|---|---|
| Sections 1–3 | Data loading, EDA, pre-processing |
| Sections 4–5 | Visualisation (8 figures) |
| Sections 6–7 | Feature justification + dataset limitations |
| Task A | MLP baseline, training, evaluation |
| Task A Optimisation | 5-phase hyperparameter search (17 experiments) |
| Task B | HOG extraction, ensemble pipeline, results |
| Task C | CNN architecture, training, evaluation |
| Task D | Clustering, k-selection, Hungarian evaluation |
| Final | Cross-task comparison and conclusion |

---

## Ethical and Social Impact

**Employment impact:** Automated defect detection directly displaces semi-skilled visual inspection roles. Responsible deployment requires worker retraining programmes and transparent communication with affected staff — not just accuracy targets.

**Domain shift risk:** The NEU dataset reflects controlled laboratory conditions on one steel grade. A model trained here may underperform systematically on different grades, lighting conditions, or camera setups. Periodic revalidation against real production samples is essential before trusting reported accuracy figures in deployment.

**Reliability and human oversight:** 95.28% accuracy means approximately 1 in 20 images is misclassified. At typical production-line throughput this corresponds to hundreds of errors per hour. AI should augment, not replace, human inspection for safety-critical decisions, with mandatory human review of low-confidence predictions.

**Safety:** Steel is used in bridges, aircraft, automotive structures, and medical equipment. A missed defect classification carries downstream safety implications. Defect type alone is insufficient — severity, location, and application context must inform accept/reject decisions.

**Explainability:** The CNN is a black box. Industrial deployment should incorporate Grad-CAM or SHAP visualisations to satisfy ISO 9001 quality documentation requirements and support regulatory audit trails. The stacking ensemble (Task B) provides interpretable feature importances and is preferable in audit-constrained environments.

---

## Future Improvements

| Enhancement | Approach | Expected Impact |
|---|---|---|
| Transfer learning | Fine-tune ResNet50/EfficientNetB0 pre-trained on ImageNet | Significant accuracy gain on 1,440 samples |
| Augmentation expansion | Elastic distortions, grid distortions, random erasing | Improved generalisation to production variation |
| Grad-CAM explainability | Gradient-weighted class activation maps on CNN | Regulatory compliance + defect localisation |
| Object detection | YOLOv8 / Faster R-CNN for bounding box output | Defect localisation, not just classification |
| Multi-scale HOG | Pyramid HOG across 3 resolutions | Better descriptor for multi-scale defects |
| Semi-supervised learning | Label propagation on Task D clusters into Task B | Exploits unlabelled data to boost supervised accuracy |
| Real-time deployment | TensorFlow Lite / ONNX export | Production-line inference at 30+ FPS on embedded hardware |

---

## Project Impact

> *Copy-ready for LinkedIn Featured Section, CV Projects, or Portfolio Website*

---

**SteelVision AI — Industrial Defect Detection | TensorFlow · scikit-learn · OpenCV · Python**

Built a complete multi-paradigm AI research pipeline for automated steel surface defect classification using the NEU Surface Defect Database, comparing four machine learning approaches from first principles to production-ready performance.

Designed and trained a 4-block Convolutional Neural Network achieving 95.28% test accuracy and 0.9528 Macro F1-Score on six defect classes from only 1,440 training images — outperforming a HOG-feature stacking ensemble (86.94%) and a 5-phase-optimised MLP (65.83%). Implemented the complete supervised learning journey from a constrained 601k-parameter MLP baseline through handcrafted HOG + PCA ensemble methods to end-to-end CNN feature learning, quantifying exactly how spatial inductive bias drives industrial vision performance.

Extended the analysis with unsupervised clustering (KMeans, Agglomerative, GMM) using Hungarian algorithm label matching and composite internal validation metrics, proving that the HOG+PCA feature space encodes genuine class-discriminative structure even without supervision — while precisely quantifying the cost of removing label information (approximately 30pp accuracy gap).

Delivered 20+ publication-quality visualisations, a comprehensive ethical impact analysis, and actionable industrial deployment recommendations. The project demonstrates end-to-end applied AI competency across computer vision, feature engineering, ensemble methods, deep learning, and unsupervised learning within a single coherent industrial problem domain.

---

## ATS Keywords

```
Python | TensorFlow | Keras | scikit-learn | OpenCV | NumPy | Pandas | Matplotlib | Seaborn
Convolutional Neural Network | CNN | Deep Learning | Computer Vision | Image Classification
Multilayer Perceptron | MLP | Neural Network | Feedforward Network
Ensemble Learning | Random Forest | Extra Trees | Gradient Boosting | Stacking Classifier
HOG | Histogram of Oriented Gradients | Feature Engineering | Feature Extraction
Principal Component Analysis | PCA | Dimensionality Reduction | StandardScaler
KMeans | Agglomerative Clustering | Gaussian Mixture Model | GMM | Unsupervised Learning
Hungarian Algorithm | Adjusted Rand Index | ARI | Normalised Mutual Information | NMI
BatchNormalization | Dropout | Data Augmentation | EarlyStopping | ReduceLROnPlateau
Adam Optimizer | RandomizedSearchCV | GridSearchCV | Cross-Validation | Stratified KFold
Defect Detection | Surface Inspection | Quality Control | Industry 4.0 | Smart Manufacturing
Steel Manufacturing | NEU Surface Defect | Industrial AI | Visual Inspection
Texture Classification | Spatial Feature Learning | t-SNE | Silhouette Score
Precision | Recall | F1-Score | Confusion Matrix | Classification Report
Machine Learning | Applied AI | Data Science | Computer Vision Engineer
Deep Learning Engineer | ML Engineer | AI Engineer | Research Engineer
Jupyter Notebook | Git | GitHub | Reproducible Research | Scientific Computing
```

---

## References

Dalal, N. and Triggs, B. (2005) Histograms of Oriented Gradients for Human Detection. CVPR 2005.

Geurts, P., Ernst, D. and Wehenkel, L. (2006) Extremely Randomised Trees. Machine Learning, 63(1).

Ioffe, S. and Szegedy, C. (2015) Batch Normalization: Accelerating Deep Network Training. ICML 2015.

Kingma, D.P. and Ba, J. (2015) Adam: A Method for Stochastic Optimization. ICLR 2015.

LeCun, Y. et al. (1998) Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE.

Mu, X., Yao, D. and Sun, H. (2018) NEU Surface Defect Database. Kaggle.

Pedregosa, F. et al. (2011) Scikit-learn: Machine Learning in Python. JMLR, 12.

Rousseeuw, P.J. (1987) Silhouettes: A Graphical Aid to Cluster Analysis. Journal of Computational and Applied Mathematics.

Srivastava, N. et al. (2014) Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15.

Van der Maaten, L. and Hinton, G. (2008) Visualizing Data using t-SNE. JMLR, 9.

Wolpert, D.H. (1992) Stacked Generalization. Neural Networks, 5(2).

---

*Four paradigms · One dataset · A clear answer on what drives industrial AI performance*

