# H9DLGA Repeat — Galaxy Morphology with CNN + VAE

**Student:** Marco Aurélio Minozzo (`x24153362`)  
**Module:** H9DLGA Deep Learning and Generative AI (Repeat)  
**Institution:** National College of Ireland

## Research question

Given class imbalance in Galaxy10 DECaLS, do VAE-generated synthetic images improve CNN morphological classification on minority classes without degrading majority-class performance?

## Highlights

| Setting | Accuracy | Macro-F1 |
|---|---|---|
| Real-only ResNet50 | 67.08% | 0.648 |
| Real + VAE synthetics | **71.51%** | **0.686** |

## Repository layout

```
src/                 Training / evaluation pipeline (CRISP-DM)
notebooks/           Walkthrough notebook
paper/               IEEE report + Q2 review sources and PDFs
presentation/        Slide outline
outputs/figures/     Evaluation figures
outputs/metrics/     Held-out metrics (JSON)
requirements.txt
```

Large artefacts are **not** in git (see `.gitignore`): Galaxy10 `.h5`, `.pt` checkpoints, and `Material de apoio/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

```bash
python -m src.run_pipeline
# or
python -m src.run_pipeline --skip-download --resume-vae
```

## Submission artefacts

- Report PDF: `paper/H9DLGA_Repeat_x24153362_Marco_Aurelio_Minozzo_Report.pdf`
- Code zip: `H9DLGA_Repeat_x24153362_Marco_Aurelio_Minozzo_Code.zip`
