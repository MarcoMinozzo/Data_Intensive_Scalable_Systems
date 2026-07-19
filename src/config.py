"""
Project configuration for Galaxy10 DECaLS CNN + VAE study.

Research question:
  Given class imbalance in Galaxy10 DECaLS, do VAE-generated synthetic images
  improve CNN morphological classification on minority classes without
  degrading majority-class performance?
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"

GALAXY10_H5_NAME = "Galaxy10_DECals.h5"
GALAXY10_H5_PATH = DATA_RAW / GALAXY10_H5_NAME

# Official Galaxy10 DECaLS release (astroNN / henrysky Galaxy10)
# Source: https://astronn.readthedocs.io/en/latest/galaxy10.html
GALAXY10_URL = (
    "https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"
)

# ---------------------------------------------------------------------------
# Dataset metadata (Galaxy10 DECaLS)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Disturbed",
    1: "Merging",
    2: "Round Smooth",
    3: "In-between Round Smooth",
    4: "Cigar Shaped Smooth",  # minority class (~334 images)
    5: "Barred Spiral",
    6: "Unbarred Tight Spiral",
    7: "Unbarred Loose Spiral",
    8: "Edge-on without Bulge",
    9: "Edge-on with Bulge",
}
NUM_CLASSES = 10
MINORITY_CLASS_IDS = [4]  # primary imbalance focus

# ---------------------------------------------------------------------------
# Preprocessing / splits
# ---------------------------------------------------------------------------
IMAGE_SIZE_VAE = 64  # practical size for ConvVAE training
IMAGE_SIZE_CNN = 224  # ResNet50 input size (Lab 2 transfer learning)
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# VAE hyperparameters (loss form matches Week 6 / VAE labs: recon + KL)
# ---------------------------------------------------------------------------
VAE_LATENT_DIM = 32
VAE_BATCH_SIZE = 64
VAE_EPOCHS = 30
VAE_LR = 1e-3
VAE_BETA = 1.0  # KL weight (beta-VAE style; 1.0 = standard ELBO)

# ---------------------------------------------------------------------------
# CNN hyperparameters (ResNet50 transfer learning, aligned with Lab 2)
# ---------------------------------------------------------------------------
CNN_BATCH_SIZE = 32
CNN_HEAD_EPOCHS = 5
CNN_FINETUNE_EPOCHS = 5
CNN_LR_HEAD = 1e-3
CNN_LR_FINETUNE = 1e-4

# How many synthetic images to add per class when balancing with VAE samples.
# None => raise each training class count up to the max class count in train set.
SYNTHETIC_TARGET_PER_CLASS = None
