# IEEE Paper Outline (H9DLGA Repeat)

> Working outline for the 8–10 page IEEE double-column report.  
> Fill results after running `python -m src.run_pipeline`.

**Working title:** Improving Galaxy Morphology Classification under Class Imbalance with VAE-Based Generative Augmentation

## Abstract (150–250 words)
- Problem: morphological classification + imbalance in Galaxy10 DECaLS
- Method: ConvVAE synthetics + ResNet50 CNN (CRISP-DM)
- Key result: TBD (macro-F1 / minority-class F1 real-only vs real+VAE)
- Contribution: empirical comparison of generative augmentation for astronomy imagery using module methods

## I. Introduction
- Motivation (survey scale; costly manual Galaxy Zoo labels)
- Research question (minority-class performance without majority degradation)
- Brief CRISP-DM overview of the study
- Paper structure (1–2 lines)

## II. Related Work
- Galaxy morphology / Galaxy Zoo / Galaxy10 DECaLS
- CNNs and transfer learning for astronomical images
- VAEs for generative modelling and data augmentation
- Critical evaluation: what prior work answers / does not answer regarding imbalance + synthetics
- Target: ~12–20 IEEE references, prioritising `Material de apoio` sources (Dive into Deep Learning, Aggarwal, module slides/labs, Alammar where relevant) plus Galaxy10/astroNN citations

## III. Deep Learning and Generative AI Methodology (CRISP-DM)
### A. Business Understanding
- Research question and success criteria (macro-F1; class-4 recall/F1)

### B. Data Understanding
- Galaxy10 DECaLS (17 736 images, 10 classes)
- Class imbalance (esp. Cigar Shaped Smooth)

### C. Data Preparation
- Stratified 70/15/15 split
- VAE resize 64×64; CNN resize 224×224 + ImageNet normalisation

### D. Modeling
- ConvVAE (lab loss: reconstruction + KL; convolutional adaptation for RGB)
- Synthetic balancing via class-conditional latent sampling
- ResNet50 transfer learning (Lab 2: head training → fine-tune layer4+FC)
- Treatments: CNN real-only vs CNN real+VAE

### E. Evaluation protocol
- Held-out test set; accuracy, macro/weighted F1, per-class metrics, confusion matrices

## IV. Evaluation / Results
- Dataset statistics and imbalance
- VAE reconstructions / sample synthetics (figures from `outputs/figures/`)
- Table: real-only vs real+VAE overall metrics
- Table/figure: per-class F1 with minority emphasis
- Discussion: implications, parameter choices, what results show / do not show

## V. Conclusions and Future Work
- Partial answer to RQ
- Limitations (VAE fidelity at 64×64; upscaling; compute)
- Extensions: conditional VAE/GAN, longer fine-tuning, multi-survey generalisation

## References
IEEE style (not Harvard).

## Figures / tables to generate from code
1. Class distribution (train split)
2. VAE reconstructions
3. Example synthetic minority-class images
4. Confusion matrices (real-only / real+VAE)
5. Per-class F1 comparison bar chart
