"""
Batch ESD analyzer for processing multiple models from HuggingFace.
Uses the optimized net_esd implementation with multi-GPU support.
"""
import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import torch
import psutil
from tqdm import tqdm

# Add parent directory to path to import net_esd
sys.path.insert(0, str(Path(__file__).parent.parent))
from net_esd import net_esd_estimator

import config
from model_utils import (
    load_model_smart,
    safe_model_cleanup,
    get_model_size_gb,
    detect_model_type
)
from gputracker.gpu_scheduler import GPUScheduledProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model to process."""
    full_name: str
    source_model: Optional[str] = None
    base_model_relation: Optional[str] = None
    index: int = 0

    @property
    def is_adapter(self) -> bool:
        """Check if this is an adapter model."""
        if self.base_model_relation:
            return self.base_model_relation.lower() in config.ADAPTER_RELATIONS
        return False

    @property
    def safe_name(self) -> str:
        """Get filesystem-safe model name."""
        return self.full_name.replace("/", "--").replace("\\", "--")


@dataclass
class ESDResult:
    """Results from ESD analysis."""
    model_info: ModelInfo
    metrics: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    processing_time: float = 0.0
    model_size_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        result = asdict(self.model_info)
        result.update({
            'success': self.success,
            'error': self.error,
            'processing_time': self.processing_time,
            'model_size_gb': self.model_size_gb,
        })

        # Add averaged metrics
        if self.metrics:
            for key, values in self.metrics.items():
                if key in ['longname', 'eigs']:
                    continue
                if isinstance(values, list) and len(values) > 0:
                    # Calculate mean, excluding None values
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        result[f"{key}_mean"] = np.mean(valid_values)
                        result[f"{key}_std"] = np.std(valid_values)
                        result[f"{key}_min"] = np.min(valid_values)
                        result[f"{key}_max"] = np.max(valid_values)

        return result


class BatchESDAnalyzer:
    """Batch analyzer for running ESD on multiple models."""

    def __init__(
        self,
        output_dir: str = "./esd_results",
        fix_fingers: Optional[str] = config.DEFAULT_FIX_FINGERS,
        evals_thresh: float = config.EVALS_THRESH,
        bins: int = config.BINS,
        filter_zeros: bool = config.FILTER_ZEROS,
        parallel: bool = config.PARALLEL_COMPUTE,
        device_ids: Optional[List[int]] = config.DEFAULT_GPU_IDS,
        max_workers: Optional[int] = config.MAX_WORKERS,
        checkpoint_frequency: int = config.CHECKPOINT_FREQUENCY,
        # GPU scheduling parameters
        gpu_pool: Optional[List[int]] = None,
        max_total_gpus: Optional[int] = None,
        gpus_per_model: int = 1,
        use_gpu_scheduling: bool = False,
    ):
        """
        Initialize the batch analyzer.

        Args:
            output_dir: Directory to save results
            fix_fingers: Method for xmin selection
            evals_thresh: Threshold for filtering eigenvalues
            bins: Number of bins for histogram
            filter_zeros: Whether to filter near-zero eigenvalues
            parallel: Use parallel GPU computation
            device_ids: GPU IDs to use for computation
            max_workers: Maximum parallel workers
            checkpoint_frequency: Save checkpoint every N models
            gpu_pool: List of GPU IDs we're allowed to use (e.g., [0,1,2,3,4,5,6,7])
            max_total_gpus: Maximum number of GPUs we can use at once from the pool
            gpus_per_model: Number of GPUs to allocate per model
            use_gpu_scheduling: Enable dynamic GPU scheduling
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ESD parameters
        self.fix_fingers = fix_fingers
        self.evals_thresh = evals_thresh
        self.bins = bins
        self.filter_zeros = filter_zeros
        self.parallel = parallel
        self.device_ids = device_ids
        self.max_workers = max_workers

        # GPU scheduling
        self.use_gpu_scheduling = use_gpu_scheduling
        self.gpu_scheduler = None
        if use_gpu_scheduling:
            self.gpu_scheduler = GPUScheduledProcessor(
                gpu_pool=gpu_pool,
                max_total_gpus=max_total_gpus,
                gpus_per_model=gpus_per_model,
                memory_threshold=500,
                max_checks=10,
            )
            logger.info(f"GPU scheduling enabled: pool={gpu_pool}, max_gpus={max_total_gpus}, per_model={gpus_per_model}")

        # Checkpoint settings
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_file = self.output_dir / "checkpoint.csv"

        # Results storage
        self.results: List[ESDResult] = []

        # Setup directories
        self.csv_dir = self.output_dir / "per_model_csv"
        self.csv_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Setup logging
        log_file = self.logs_dir / f"batch_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def process_model(self, model_info: ModelInfo) -> ESDResult:
        """
        Process a single model.

        Args:
            model_info: Model information

        Returns:
            ESDResult with metrics or error
        """
        start_time = time.time()
        result = ESDResult(model_info=model_info)

        # Check if already processed
        csv_path = self.csv_dir / f"{model_info.safe_name}.csv"
        if csv_path.exists() and not getattr(self, 'overwrite', False):
            logger.info(f"Skipping {model_info.full_name} - already processed")
            try:
                # Load existing results
                df = pd.read_csv(csv_path)
                result.metrics = df.to_dict('list') # type: ignore
                result.success = True
                result.processing_time = 0
                return result
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")

        model = None
        allocated_gpus = None
        try:
            # Allocate GPUs if scheduling is enabled
            if self.use_gpu_scheduling:
                allocated_gpus = self.gpu_scheduler.allocate_gpus() # type: ignore
                # Use allocated GPUs for this model
                device_ids_to_use = allocated_gpus
                logger.info(f"Allocated GPUs {allocated_gpus} for {model_info.full_name}")
            else:
                device_ids_to_use = self.device_ids
            
            # Check RAM availability
            if psutil.virtual_memory().available / 1e9 < config.RAM_THRESHOLD_GB:
                logger.warning(f"Low RAM, waiting before processing {model_info.full_name}")
                time.sleep(10)

            # Load model
            logger.info(f"Loading model: {model_info.full_name}")
            model = load_model_smart(
                model_name=model_info.full_name,
                model_type="adapter" if model_info.is_adapter else "base",
                base_model=model_info.source_model if model_info.is_adapter else None,
            )

            # Get model size
            result.model_size_gb = get_model_size_gb(model)
            logger.info(f"Model size: {result.model_size_gb:.2f} GB")

            # Run ESD analysis using the fast implementation
            logger.info(f"Running ESD analysis for {model_info.full_name}")
            metrics = net_esd_estimator(
                net=model,
                EVALS_THRESH=self.evals_thresh,
                bins=self.bins,
                fix_fingers=None if self.fix_fingers == "DKS" else self.fix_fingers,
                filter_zeros=self.filter_zeros,
                parallel=self.parallel,
                device_ids=device_ids_to_use,  # Use allocated or default GPUs
                max_workers=self.max_workers,
            )

            result.metrics = metrics
            result.success = True

            # Save per-model results
            self._save_model_results(model_info, metrics, csv_path)

            logger.info(f"Successfully processed {model_info.full_name}")

        except Exception as e:
            logger.error(f"Failed to process {model_info.full_name}: {e}")
            result.success = False
            result.error = str(e)

        finally:
            # Clean up model
            safe_model_cleanup(model)
            
            # Release allocated GPUs if scheduling is enabled
            if self.use_gpu_scheduling and allocated_gpus:
                self.gpu_scheduler.release_gpus(allocated_gpus) # type: ignore
                logger.info(f"Released GPUs {allocated_gpus} for {model_info.full_name}")
            
            result.processing_time = time.time() - start_time

        return result

    def _save_model_results(
        self,
        model_info: ModelInfo,
        metrics: Dict[str, Any],
        csv_path: Path
    ) -> None:
        """Save results for a single model to CSV."""
        # Prepare data for DataFrame
        data = {key: values for key, values in metrics.items() if key != 'eigs'}

        # Add summary row with averaged values
        summary_row = {}
        for key, values in data.items():
            if key == 'longname':
                summary_row[key] = 'SUMMARY'
            elif key == 'params' and isinstance(values, list) and len(values) > 0:
                # Sum total parameters
                summary_row[key] = sum(v for v in values if v is not None)
            elif isinstance(values, list) and len(values) > 0:
                # Calculate mean for other metrics
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    summary_row[key] = np.mean(valid_values)
                else:
                    summary_row[key] = None
            else:
                summary_row[key] = None

        # Append summary row
        for key in data:
            if isinstance(data[key], list):
                data[key].append(summary_row.get(key))

        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logger.debug(f"Saved results to {csv_path}")

    def process_batch(
        self,
        models: List[ModelInfo],
        num_workers: int = 1,
        overwrite: bool = False
    ) -> pd.DataFrame:
        """
        Process a batch of models.

        Args:
            models: List of models to process
            num_workers: Number of parallel workers
            overwrite: Whether to overwrite existing results

        Returns:
            DataFrame with results for all models
        """
        self.overwrite = overwrite
        self.results = []

        logger.info(f"Processing {len(models)} models with {num_workers} workers")

        if num_workers == 1:
            # Sequential processing
            for model_info in tqdm(models, desc="Processing models"):
                result = self.process_model(model_info)
                self.results.append(result)

                # Save checkpoint
                if len(self.results) % self.checkpoint_frequency == 0:
                    self._save_checkpoint()
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self.process_model, model_info): model_info
                    for model_info in models
                }

                for future in tqdm(as_completed(futures), total=len(models), desc="Processing models"):
                    result = future.result()
                    self.results.append(result)

                    # Save checkpoint
                    if len(self.results) % self.checkpoint_frequency == 0:
                        self._save_checkpoint()

        # Final save
        df = self._create_summary_dataframe()
        self._save_final_results(df)

        return df

    def _save_checkpoint(self) -> None:
        """Save intermediate checkpoint."""
        df = self._create_summary_dataframe()
        df.to_csv(self.checkpoint_file, index=False)
        logger.info(f"Saved checkpoint with {len(self.results)} models")

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        rows = [result.to_dict() for result in self.results]
        return pd.DataFrame(rows)

    def _save_final_results(self, df: pd.DataFrame) -> None:
        """Save final results."""
        # Save main results
        output_file = self.output_dir / "esd_results_summary.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved final results to {output_file}")

        # Save failed models separately
        failed = df[df['success'] == False]
        if not failed.empty:
            failed_file = self.output_dir / "failed_models.csv"
            failed.to_csv(failed_file, index=False)
            logger.warning(f"Found {len(failed)} failed models, saved to {failed_file}")

        # Print summary statistics
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Total models: {len(df)}")
        print(f"Successful: {df['success'].sum()}")
        print(f"Failed: {(~df['success']).sum()}")

        if 'alpha_mean' in df.columns:
            print(f"\nAlpha statistics:")
            print(f"  Mean: {df['alpha_mean'].mean():.3f}")
            print(f"  Std:  {df['alpha_mean'].std():.3f}")
            print(f"  Min:  {df['alpha_mean'].min():.3f}")
            print(f"  Max:  {df['alpha_mean'].max():.3f}")

        print(f"\nResults saved to: {output_file}")
        print("="*60)


def load_models_from_csv(csv_path: str) -> List[ModelInfo]:
    """
    Load model information from CSV file.

    Args:
        csv_path: Path to CSV file with model information

    Returns:
        List of ModelInfo objects
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    required = ['full_name']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    models = []
    for idx, row in df.iterrows():
        model_info = ModelInfo(
            full_name=row['full_name'],
            source_model=row.get('source_model'),
            base_model_relation=row.get('base_model_relation'),
            index=idx # type: ignore
        )
        models.append(model_info)

    return models
