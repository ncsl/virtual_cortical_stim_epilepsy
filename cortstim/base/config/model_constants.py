# coding=utf-8

# frequency bands
from enum import Enum
DALPHA = [0, 15]
BETA = [15, 30]
GAMMA = [30, 90]
HIGH = [90, 200]

''' TVB SIMULATION '''
# how can we inject noise into our models?
WHITE_NOISE = "White"
COLORED_NOISE = "Colored"
NOISE_SEED = 42

TIME_DELAYS_FLAG = 0.0
MAX_DISEASE_VALUE = 1.0 - 10 ** -3


# coding=utf-8

# Default model parameters

''' FRAGILITY MODEL '''
# MODEL params in milliseconds
WINSIZE_LTV = 250
STEPSIZE_LTV = 125
RADIUS = 1.5  # perturbation radius
PERTURBTYPE = 'C'
COLUMN_PERTURBATION = 'c'
ROW_PERTURBATION = 'r'

''' FREQUENCY MODEL '''
# Default model parameters
WINSIZE_SPEC = 1000
STEPSIZE_SPEC = 500
MTBANDWIDTH = 4  # multitaper FFT bandwidth
WAVELETWIDTH = 6  # WIDTH of our wavelets


DATASET_TYPES = ['scalp', 'ieeg']


deltaband = [0, 4]
thetaband = [4, 8]
alphaband = [8, 13]
betaband = [13, 30]
gammaband = [30, 90]
highgammaband = [90, 500]


class Freqbands(Enum):
    DELTA = deltaband
    THETA = thetaband
    ALPHA = alphaband
    BETA = betaband
    GAMMA = gammaband
    HIGH = highgammaband


''' DEEP LEARNING MODEL '''
# coding=utf-8
# import torch

# Default model parameters
LEARNING_RATE = 1e-4
DROPOUT = True
SHUFFLE = True
AUGMENT = True

NUM_EPOCHS = 100  # 200 # 300
BATCH_SIZE = 64  # 32

# seq to seq model parameters
MAX_LENGTH = 250  # length of each window
USE_CUDA = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# frequency bands
DALPHA = [0, 15]
BETA = [15, 30]
GAMMA = [30, 90]
HIGH = [90, 200]

# Dataset constants
TRAIN = 'TRAIN'
TEST = 'TEST'
VALIDATE = 'VALIDATE'

# how can we inject noise into our models?
NOISE_SEED = 42
