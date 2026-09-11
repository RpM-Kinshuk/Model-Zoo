"""Constants and configuration for ESD analysis."""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keys for result dictionary returned by net_esd_estimator
RESULT_KEYS = [
    'D', 'M', 'N', 'alpha', 'alpha_weighted', 'entropy', 'log_alpha_norm',
    'log_norm', 'log_spectral_norm', 'longname', 'matrix_rank', 'norm',
    'num_evals', 'spectral_norm', 'stable_rank', 'xmax', 'xmin', 'fit_xmin',
    'n_tail', 'params', 'eigs', 'fit_status', 'raw_num_evals', 'raw_norm',
    'raw_spectral_norm', 'raw_matrix_rank', 'raw_entropy', 'evals_thresh',
    'filter_zeros', 'source_dtype', 'compute_dtype', 'weight_layout',
    'module_name', 'slice', 'compute_device'
]
