# Dream Encoding Using Sentiment Analysis

This repository presents the full implementation of my Master's Thesis project, which explores the intersection of Natural Language Processing (NLP), sentiment analysis, and psychology. The goal is to develop a robust classification pipeline that decodes emotional content from dream descriptions using machine learning techniques.

The methodology integrates handcrafted and statistical text features, advanced resampling strategies to address class imbalance, and neural network modeling. The system achieves high accuracy on a complex, imbalanced, and subjective dataset composed of free-form dream reports.

---

## Objective

To investigate whether emotional valence and sentiment can be reliably extracted from free-text dream descriptions, and to evaluate the effectiveness of hybrid features (statistical + semantic) in multi-class sentiment classification settings.

---

## Dataset Overview

- **Domain:** Free-form dream descriptions
- **Labels:** 7-point sentiment scale (1 to 7), annotated by:
  - Dreamers
  - Independent judges
  - External annotators (Nathan dataset)
- **Challenges:**
  - Strong class imbalance
  - Subjectivity and metaphor in language
  - Limited sample size

---

## Methodology

### 1. Preprocessing
- Text normalization
- Stopword removal and lemmatization (NLTK)
- Tokenization and cleanup

### 2. Feature Extraction
- **TF-IDF**: 1000-dimensional vector (unigrams + bigrams)
- **Empath**: 194 lexical-semantic categories
- Feature concatenation and standardization (StandardScaler)

### 3. Class Imbalance Handling
- **SMOTEENN**: Combined oversampling and cleaning
- Manual class-wise SMOTE + duplication for minority enhancement

### 4. Classification Model
- **MLPClassifier** (scikit-learn)
  - Hidden layers: 1 × 100 neurons
  - Activation: ReLU
  - Optimizer: Adam
- Evaluated using both 5-fold Cross-Validation and Leave-One-Out Cross-Validation (LOOCV)

---

## Evaluation Results

| Evaluation Strategy | Accuracy | Macro F1-score |
|---------------------|----------|----------------|
| 5-Fold Cross-Validation | 99.19% ± 0.009 | 99.11% ± 0.011 |
| LOOCV | 99.59% | 99.59% |
| Final Hold-Out Test | 97.96% | 98.0% (macro F1, approximated) |

Additional metrics include per-class precision/recall and confusion matrices.

---

## Cross-Annotation Correlation

The model was further evaluated across labels from three annotation groups:
- **Dreamers**
- **Judges**
- **Nathan (external annotator)**

**Cohen's Kappa** and **Spearman's Rank Correlation** were used to measure agreement. Findings suggest strong alignment between Nathan and Judges, but weaker correlation with Dreamer-provided labels.

---

## Repository Structure

