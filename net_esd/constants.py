"""Constants and configuration for ESD analysis."""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Numerical stability epsilon for entropy calculations
# EPSILON = 1e-10
EPSILON = 6e-05

# Keys for result dictionary returned by net_esd_estimator
RESULT_KEYS = [
    'D', 'M', 'N', 'alpha', 'alpha_weighted', 'entropy', 'log_alpha_norm',
    'log_norm', 'log_spectral_norm', 'longname', 'matrix_rank', 'norm',
    'num_evals', 'spectral_norm', 'stable_rank', 'xmax', 'xmin', 'params', 'eigs'
]
