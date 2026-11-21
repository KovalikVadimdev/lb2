"""Central settings of the project.

All numbers that define the experiments live here, so that they can be
changed in one place. The experiment scripts and the library read them
from this module.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Data: the 8x8 handwritten digits shipped with scikit-learn (1797 images,
# 10 classes). Pixel values range from 0 to 16.
IMAGE_SIZE = 8  # The image side in pixels; an image has IMAGE_SIZE ** 2 pixels.
N_CLASSES = 10
# Pixels above this value become +1 (ink), the others -1 (background) when an
# image is turned into a bipolar pattern for the Hopfield network.
BINARIZE_THRESHOLD = 8

# Share of the samples that goes to the test set. The split is stratified
# (both sets keep the share of every class). The same seed gives the same
# split, noise and initial weights, so the results can be reproduced.
TEST_SIZE = 0.3
RANDOM_SEED = 42

# Hopfield network
# Learning rules: "hebb" (outer products of the patterns) and "pseudoinverse"
# (projection rule, stores more patterns without crosstalk).
HOPFIELD_RULES = ("hebb", "pseudoinverse")
HOPFIELD_MAX_ITERATIONS = 100  # Recall stops earlier if the state stops changing.
# Share of the pixels flipped in the input image (0.0 = no noise).
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3)

# Early stopping (Elman network and the baseline): training stops when the
# training loss has not improved by at least MIN_DELTA for PATIENCE
# consecutive epochs, or after MAX_EPOCHS.
MAX_EPOCHS = 2000
PATIENCE = 10
MIN_DELTA = 0.0001

# Training
BATCH_SIZE = 32  # Samples per weight update (mini-batch).
DEFAULT_LEARNING_RATE = 0.01
MOMENTUM = 0.9  # Share of the previous step kept by SGD with momentum.
TIMING_REPEATS = 20  # Repeated measurements of the classification time.
# Repeats of a training whose time is averaged in the main experiments.
N_REPEATS = 5

# Elman network: the image is read row by row, so a sequence has IMAGE_SIZE
# steps with IMAGE_SIZE pixels each. The context layer has as many neurons
# as the hidden layer.
ELMAN_HIDDEN_UNITS = 16
ELMAN_OPTIMIZERS = ("sgd", "adam")
# Gradient clipping limit for backpropagation through time; keeps the
# gradient from growing too large over the steps.
ELMAN_GRAD_CLIP = 5.0

# Elman network: learning rates tried for every optimizer.
ELMAN_LEARNING_RATES = {"sgd": (0.01, 0.1), "adam": (0.001, 0.01)}
# Elman network: hidden layer sizes of the size study, besides the default
# ELMAN_HIDDEN_UNITS; they are trained with Adam and learning rate 0.01.
ELMAN_HIDDEN_SIZES_STUDY = (4, 8, 32)

# Baseline feedforward network (multilayer perceptron) on the same data.
MLP_HIDDEN_UNITS = 16
MLP_OPTIMIZERS = ("sgd", "adam")
# Multilayer perceptron: learning rates tried for every optimizer (the same
# as for the Elman network, so that the comparison is fair).
MLP_LEARNING_RATES = {"sgd": (0.01, 0.1), "adam": (0.001, 0.01)}
# Hidden layer sizes of the size study, besides MLP_HIDDEN_UNITS.
MLP_HIDDEN_SIZES_STUDY = (4, 8, 32)
