#!/usr/bin/env python3
"""
Flexible GPU-aware experiment management system for ESD analysis.
Handles shared GPU environments with intelligent scheduling and resource management.

Usage:
    # For ESD analysis with model list
    python kinexp.py --mode esd --model-list ../atlas_metadata.csv --max-gpus 4
    
    # For single model
    python kinexp.py --mode esd --single-model openai-community/gpt2 --max-gpus 2
    
    # For custom experiments
    python kinexp.py --mode custom --commands commands.txt --max-gpus 4
"""

import os
import sys
import argparse
import pandas as pd
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gputracker.gputracker import get_logger, DispatchThread

# Set threading layer for MKL
os.environ['MKL_THREADING_LAYER'] = 'gnu'

# Default configuration
DEFAULT_CONFIG = {
    'gpu_memory_threshold': 500,  # MB
    'max_checks': 10,  # Number of checks before considering GPU free
    'check_interval': 10,  # Seconds between checks
    'num_gpus_per_job': 1,  # GPUs needed per job
    'output_dir': './esd_results',
    'log_dir': './logs',
}

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    mode: str = 'esd'  # 'esd' or 'custom'
    model_list: Optional[str] = None
    single_model: Optional[str] = None
    commands_file: Optional[str] = None
    output_dir: str = './esd_results'
    log_dir: str = './logs'
    gpu_list: List[int] = None
    max_gpus: int = 4
    gpu_memory_threshold: int = 500
    max_checks: int = 10
    num_gpus_per_job: int = 1
    overwrite: bool = False
    esd_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.gpu_list is None:
            # Auto-detect available GPUs
            import torch
            if torch.cuda.is_available():
                self.gpu_list = list(range(min(torch.cuda.device_count(), self.max_gpus)))
            else:
                self.gpu_list = []
        
        if self.esd_options is None:
            self.esd_options = {
                'fix_fingers': 'xmin_mid',
                'evals_thresh': 1e-5,
                'bins': 100,
                'no_parallel_gpu': False,
            }


class ESDExperiment:
    """Manages ESD analysis experiments with GPU tracking."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the experiment."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = f"esd_experiment_{timestamp}.log"
        return get_logger(self.config.log_dir, log_file)
    
    def generate_esd_commands(self) -> List[str]:
        """Generate ESD analysis commands for models."""
        commands = []
        
        if self.config.model_list:
            # Load models from CSV
            df = pd.read_csv(self.config.model_list)
            
            # Check if already processed
            processed_models = set()
            results_dir = Path(self.config.output_dir) / 'per_model_csv'
            if results_dir.exists() and not self.config.overwrite:
                processed_models = {f.stem for f in results_dir.glob('*.csv')}
            
            for idx, row in df.iterrows():
                model_name = row['full_name']
                safe_name = model_name.replace('/', '--')
                
                if safe_name in processed_models:
                    self.logger.info(f"Skipping already processed: {model_name}")
                    continue
                
                cmd = self._build_esd_command(row)
                commands.append(cmd)
                
        elif self.config.single_model:
            # Single model command
            row = {'full_name': self.config.single_model}
            cmd = self._build_esd_command(row)
            commands.append(cmd)
            
        self.logger.info(f"Generated {len(commands)} commands for processing")
        return commands
    
    def _build_esd_command(self, row: Dict[str, Any]) -> str:
        """Build ESD analysis command for a single model."""
        cmd_parts = [
            "OMP_NUM_THREADS=1",
            "python",
            str(Path(__file__).parent.parent / "esd_analysis" / "run_analysis.py"),
            "--single-model", row['full_name'],
            "--output-dir", self.config.output_dir,
        ]
        
        # Add base model if adapter
        if 'source_model' in row and pd.notna(row['source_model']):
            cmd_parts.extend(["--base-model", row['source_model']])
        
        # Add model type if specified
        if 'base_model_relation' in row and pd.notna(row['base_model_relation']):
            relation = row['base_model_relation'].lower()
            if relation in ['adapter', 'lora', 'peft']:
                cmd_parts.extend(["--model-type", "adapter"])
        
        # Add ESD options
        esd_opts = self.config.esd_options
        if esd_opts.get('fix_fingers'):
            cmd_parts.extend(["--fix-fingers", esd_opts['fix_fingers']])
        if esd_opts.get('evals_thresh'):
            cmd_parts.extend(["--evals-thresh", str(esd_opts['evals_thresh'])])
        if esd_opts.get('bins'):
            cmd_parts.extend(["--bins", str(esd_opts['bins'])])
        if esd_opts.get('no_parallel_gpu'):
            cmd_parts.append("--no-parallel-gpu")
        
        if self.config.overwrite:
            cmd_parts.append("--overwrite")
        
        return " ".join(cmd_parts)
    
    def load_custom_commands(self) -> List[str]:
        """Load custom commands from file."""
        with open(self.config.commands_file, 'r') as f:
            commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return commands
    
    def run(self):
        """Run the experiment with GPU tracking."""
        self.logger.info(f"Starting experiment in {self.config.mode} mode")
        self.logger.info(f"Available GPUs: {self.config.gpu_list}")
        self.logger.info(f"GPU memory threshold: {self.config.gpu_memory_threshold} MB")
        
        # Generate commands based on mode
        if self.config.mode == 'esd':
            commands = self.generate_esd_commands()
        elif self.config.mode == 'custom':
            commands = self.load_custom_commands()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        
        if not commands:
            self.logger.info("No commands to execute")
            return
        
        self.logger.info(f"Total commands to execute: {len(commands)}")
        
        # Create dispatch thread for GPU management
        dispatch_thread = DispatchThread(
            name="ESD Analysis Dispatcher",
            bash_command_list=commands,
            logger=self.logger,
            gpu_m_th=self.config.gpu_memory_threshold,
            gpu_list=self.config.gpu_list,
            maxcheck=self.config.max_checks,
            num_gpus_needed=self.config.num_gpus_per_job,
        )
        
        # Start and wait for completion
        dispatch_thread.start()
        dispatch_thread.join()
        
        # Wait a bit for cleanup
        time.sleep(5)
        
        self.logger.info("Experiment completed")
        self._print_summary()

    def _print_summary(self):
        """Print summary of the experiment."""
        results_file = Path(self.config.output_dir) / "esd_results_summary.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            self.logger.info("\n" + "="*60)
            self.logger.info("EXPERIMENT SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"Total models processed: {len(df)}")
            if 'success' in df.columns:
                self.logger.info(f"Successful: {df['success'].sum()}")
                self.logger.info(f"Failed: {(~df['success']).sum()}")
            if 'alpha_mean' in df.columns:
                self.logger.info(f"Alpha stats - Mean: {df['alpha_mean'].mean():.3f}, Std: {df['alpha_mean'].std():.3f}")
            self.logger.info(f"Results saved to: {results_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU-aware experiment management for ESD analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process models from CSV with 4 GPUs max
  python kinexp.py --mode esd --model-list ../atlas_metadata.csv --max-gpus 4
  
  # Process single model
  python kinexp.py --mode esd --single-model openai-community/gpt2 --max-gpus 2
  
  # Run custom commands
  python kinexp.py --mode custom --commands commands.txt --gpu-list 0 1 2 3
  
  # With specific GPU list and memory threshold
  python kinexp.py --mode esd --model-list models.csv --gpu-list 4 5 6 7 --gpu-mem-threshold 1000
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['esd', 'custom'],
        default='esd',
        help='Experiment mode: esd for ESD analysis, custom for custom commands'
    )
    
    # Input options
    parser.add_argument(
        '--model-list',
        type=str,
        help='CSV file with models to process (for esd mode)'
    )
    parser.add_argument(
        '--single-model',
        type=str,
        help='Single model to process (for esd mode)'
    )
    parser.add_argument(
        '--commands',
        type=str,
        help='File with custom commands to run (for custom mode)'
    )
    
    # GPU configuration
    parser.add_argument(
        '--gpu-list',
        type=int,
        nargs='+',
        help='List of GPU IDs to use (e.g., --gpu-list 0 1 2 3)'
    )
    parser.add_argument(
        '--max-gpus',
        type=int,
        default=4,
        help='Maximum number of GPUs to use (default: 4)'
    )
    parser.add_argument(
        '--gpu-mem-threshold',
        type=int,
        default=500,
        help='GPU memory threshold in MB to consider GPU free (default: 500)'
    )
    parser.add_argument(
        '--max-checks',
        type=int,
        default=10,
        help='Number of checks before considering GPU free (default: 10)'
    )
    parser.add_argument(
        '--gpus-per-job',
        type=int,
        default=1,
        help='Number of GPUs needed per job (default: 1)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./esd_results',
        help='Output directory for results (default: ./esd_results)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for log files (default: ./logs)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results'
    )
    
    # ESD specific options
    parser.add_argument(
        '--fix-fingers',
        choices=['xmin_mid', 'xmin_peak', 'DKS'],
        default='xmin_mid',
        help='ESD xmin selection method'
    )
    parser.add_argument(
        '--evals-thresh',
        type=float,
        default=1e-5,
        help='Threshold for filtering eigenvalues'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=100,
        help='Number of bins for histogram'
    )
    parser.add_argument(
        '--no-parallel-gpu',
        action='store_true',
        help='Disable parallel GPU computation within ESD'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate arguments
    if args.mode == 'esd':
        if not (args.model_list or args.single_model):
            print("Error: ESD mode requires --model-list or --single-model")
            sys.exit(1)
    elif args.mode == 'custom':
        if not args.commands:
            print("Error: Custom mode requires --commands file")
            sys.exit(1)
    
    # Create configuration
    config = ExperimentConfig(
        mode=args.mode,
        model_list=args.model_list,
        single_model=args.single_model,
        commands_file=args.commands,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        gpu_list=args.gpu_list,
        max_gpus=args.max_gpus,
        gpu_memory_threshold=args.gpu_mem_threshold,
        max_checks=args.max_checks,
        num_gpus_per_job=args.gpus_per_job,
        overwrite=args.overwrite,
        esd_options={
            'fix_fingers': args.fix_fingers,
            'evals_thresh': args.evals_thresh,
            'bins': args.bins,
            'no_parallel_gpu': args.no_parallel_gpu,
        }
    )
    
    # Run experiment
    experiment = ESDExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()