#!/usr/bin/env python3
"""
Command-line interface for running batch ESD analysis on HuggingFace models.

Usage:
    python run_analysis.py --model-list atlas_metadata.csv --output-dir ./results
    python run_analysis.py --model-list models.csv --output-dir ./results --num-workers 4
    python run_analysis.py --single-model openai-community/gpt2 --output-dir ./results
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_analyzer import BatchESDAnalyzer, ModelInfo, load_models_from_csv
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch ESD analysis on HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process models from CSV file
  python run_analysis.py --model-list atlas_metadata.csv --output-dir ./results
  
  # Process single model
  python run_analysis.py --single-model openai-community/gpt2 --output-dir ./results
  
  # Process with multiple workers
  python run_analysis.py --model-list models.csv --num-workers 4
  
  # Process adapter model with base model specified
  python run_analysis.py --single-model peft-internal-testing/gpt2-lora-random \\
                        --base-model openai-community/gpt2 --output-dir ./results
  
  # GPU scheduling: Use max 4 GPUs from pool of 8, with 1 GPU per model
  python run_analysis.py --model-list models.csv --output-dir ./results \\
                        --use-gpu-scheduling --gpu-pool 0 1 2 3 4 5 6 7 \\
                        --max-total-gpus 4 --gpus-per-model 1
  
  # GPU scheduling: Use max 4 GPUs, with 2 GPUs per model (2 models in parallel)
  python run_analysis.py --model-list models.csv --output-dir ./results \\
                        --use-gpu-scheduling --gpu-pool 0 1 2 3 4 5 6 7 \\
                        --max-total-gpus 4 --gpus-per-model 2 --num-workers 2
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--model-list",
        type=str,
        help="Path to CSV file with model information (columns: full_name, source_model, base_model_relation)"
    )
    input_group.add_argument(
        "--single-model",
        type=str,
        help="Process a single model by name/path"
    )

    # Model specification
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model for adapter (only used with --single-model)"
    )
    parser.add_argument(
        "--model-type",
        choices=["base", "adapter"],
        help="Model type (auto-detected if not specified)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./esd_results",
        help="Directory to save results (default: ./esd_results)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results"
    )

    # ESD parameters
    parser.add_argument(
        "--fix-fingers",
        type=str,
        choices=["xmin_mid", "xmin_peak", "DKS"],
        default=config.DEFAULT_FIX_FINGERS,
        help=f"Method for xmin selection (default: {config.DEFAULT_FIX_FINGERS})"
    )
    parser.add_argument(
        "--evals-thresh",
        type=float,
        default=config.EVALS_THRESH,
        help=f"Threshold for filtering eigenvalues (default: {config.EVALS_THRESH})"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=config.BINS,
        help=f"Number of bins for histogram (default: {config.BINS})"
    )
    parser.add_argument(
        "--no-filter-zeros",
        action="store_true",
        help="Do not filter near-zero eigenvalues"
    )

    # Computation options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for batch processing (default: 1)"
    )
    parser.add_argument(
        "--no-parallel-gpu",
        action="store_true",
        help="Disable parallel GPU computation for ESD"
    )
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="+",
        help="GPU device IDs to use (e.g., --device-ids 0 1 2)"
    )
    parser.add_argument(
        "--max-gpu-workers",
        type=int,
        help="Maximum workers for parallel GPU computation"
    )
    
    # GPU scheduling options
    parser.add_argument(
        "--use-gpu-scheduling",
        action="store_true",
        help="Enable dynamic GPU scheduling for batch processing"
    )
    parser.add_argument(
        "--gpu-pool",
        type=int,
        nargs="+",
        help="GPU pool to use for scheduling (e.g., --gpu-pool 0 1 2 3 4 5 6 7)"
    )
    parser.add_argument(
        "--max-total-gpus",
        type=int,
        help="Maximum total GPUs to use from the pool (e.g., --max-total-gpus 4)"
    )
    parser.add_argument(
        "--gpus-per-model",
        type=int,
        default=1,
        help="Number of GPUs to allocate per model (default: 1)"
    )

    # Processing options
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of models to process (for testing)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=config.CHECKPOINT_FREQUENCY,
        help=f"Save checkpoint every N models (default: {config.CHECKPOINT_FREQUENCY})"
    )

    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output"
    )

    return parser.parse_args()


def setup_logging(args):
    """Setup logging based on arguments."""
    if args.quiet:
        level = logging.ERROR
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )


def load_models(args) -> List[ModelInfo]:
    """Load models based on arguments."""
    if args.single_model:
        # Single model mode
        model_info = ModelInfo(
            full_name=args.single_model,
            source_model=args.base_model,
            base_model_relation="adapter" if args.model_type == "adapter" else None,
            index=0
        )
        return [model_info]
    else:
        # Load from CSV
        models = load_models_from_csv(args.model_list)

        # Apply limit if specified
        if args.limit and args.limit > 0:
            models = models[:args.limit]
            logger.info(f"Limited to first {args.limit} models")

        return models


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args)

    # Validate arguments
    if args.base_model and not args.single_model:
        logger.error("--base-model can only be used with --single-model")
        sys.exit(1)

    # Load models
    try:
        models = load_models(args)
        logger.info(f"Loaded {len(models)} models to process")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)

    # Create analyzer
    analyzer = BatchESDAnalyzer(
        output_dir=args.output_dir,
        fix_fingers=args.fix_fingers,
        evals_thresh=args.evals_thresh,
        bins=args.bins,
        filter_zeros=not args.no_filter_zeros,
        parallel=not args.no_parallel_gpu,
        device_ids=args.device_ids,
        max_workers=args.max_gpu_workers,
        checkpoint_frequency=args.checkpoint_freq,
        # GPU scheduling parameters
        use_gpu_scheduling=args.use_gpu_scheduling,
        gpu_pool=args.gpu_pool,
        max_total_gpus=args.max_total_gpus,
        gpus_per_model=args.gpus_per_model,
    )

    # Process models
    try:
        logger.info(f"Starting batch processing with {args.num_workers} workers")
        results_df = analyzer.process_batch(
            models=models,
            num_workers=args.num_workers,
            overwrite=args.overwrite
        )

        # Print summary
        if not args.quiet:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"Processed {len(results_df)} models")
            print(f"Results saved to: {args.output_dir}")

            # Show some statistics
            if 'alpha_mean' in results_df.columns:
                alpha_stats = results_df['alpha_mean'].describe()
                print("\nAlpha distribution:")
                print(alpha_stats.to_string())

            # Show failed models if any
            failed = results_df[results_df['success'] == False]
            if not failed.empty:
                print(f"\n⚠️  {len(failed)} models failed to process")
                print("Failed models:")
                for _, row in failed.iterrows():
                    print(f"  - {row['full_name']}: {row.get('error', 'Unknown error')}")

        # Exit with appropriate code
        failed_count = (results_df['success'] == False).sum()
        if failed_count > 0:
            sys.exit(1)  # Partial failure
        else:
            sys.exit(0)  # Success

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Failed during processing: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    main()
