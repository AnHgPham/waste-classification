"""
Waste Classification System - Main Entry Point

This script provides a unified CLI interface to run the entire waste classification pipeline.

Usage:
    # Run full pipeline
    python main.py --all

    # Run specific week
    python main.py --week 1
    python main.py --week 2

    # Run individual tasks
    python main.py --explore
    python main.py --preprocess
    python main.py --train-baseline
    python main.py --train-transfer
    python main.py --evaluate --model mobilenetv2
    python main.py --realtime
    python main.py --optimize

    # Quick mode (fast pipeline)
    python main.py --quick

"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from src.config import *


def run_script(script_path, args=None):
    """
    Run a Python script as a subprocess.

    Arguments:
    script_path -- str, path to the script relative to project root.
    args -- list or None, additional command-line arguments.

    Returns:
    success -- bool, True if script ran successfully.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Script failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        return False


def run_week1(args):
    """Run all Week 1 tasks: Data preparation and baseline CNN."""
    print("\n" + "🎯" * 35)
    print("WEEK 1: DATA PREPARATION AND BASELINE CNN")
    print("🎯" * 35 + "\n")

    # Data exploration
    if not run_script("scripts/01_data_exploration.py"):
        return False

    # Data preprocessing
    if not run_script("scripts/02_preprocessing.py"):
        return False

    # Train baseline model
    script_args = []
    if args.epochs:
        script_args.extend(['--epochs', str(args.epochs)])

    if not run_script("scripts/03_baseline_training.py", script_args):
        return False

    # Evaluate baseline model
    if not run_script("scripts/99_evaluate_model.py", ['--model', 'baseline']):
        return False

    return True


def run_week2(args):
    """Run all Week 2 tasks: Transfer learning."""
    print("\n" + "🎯" * 35)
    print("WEEK 2: TRANSFER LEARNING WITH MOBILENETV2")
    print("🎯" * 35 + "\n")

    script_args = []
    if args.phase1_epochs:
        script_args.extend(['--phase1-epochs', str(args.phase1_epochs)])
    if args.phase2_epochs:
        script_args.extend(['--phase2-epochs', str(args.phase2_epochs)])

    # Train transfer learning model
    if not run_script("scripts/04_transfer_learning.py", script_args):
        return False

    # Evaluate transfer learning model
    if not run_script("scripts/99_evaluate_model.py", ['--model', 'mobilenetv2']):
        return False

    return True


def run_week3(args):
    """Run all Week 3 tasks: Real-time detection."""
    print("\n" + "🎯" * 35)
    print("WEEK 3: REAL-TIME WASTE DETECTION")
    print("🎯" * 35 + "\n")

    script_args = ['--model', args.model if args.model else 'mobilenetv2']

    if not run_script("scripts/05_realtime_detection.py", script_args):
        return False

    return True


def run_week4(args):
    """Run all Week 4 tasks: Model optimization."""
    print("\n" + "🎯" * 35)
    print("WEEK 4: MODEL OPTIMIZATION FOR DEPLOYMENT")
    print("🎯" * 35 + "\n")

    script_args = ['--model', args.model if args.model else 'mobilenetv2']

    if not run_script("scripts/06_model_optimization.py", script_args):
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Waste Classification System - Main CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire pipeline
  python main.py --all

  # Run specific week
  python main.py --week 1

  # Run individual tasks
  python main.py --explore
  python main.py --train-baseline --epochs 30
  python main.py --evaluate --model mobilenetv2

  # Quick mode (reduced epochs for testing)
  python main.py --quick
        """
    )

    # Week-based execution
    parser.add_argument('--all', action='store_true',
                        help='Run the entire pipeline (all 4 weeks)')
    parser.add_argument('--week', type=int, choices=[1, 2, 3, 4],
                        help='Run all tasks for a specific week')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: run full pipeline with reduced epochs')

    # Individual tasks
    parser.add_argument('--explore', action='store_true',
                        help='Run data exploration')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing')
    parser.add_argument('--train-baseline', action='store_true',
                        help='Train baseline CNN model')
    parser.add_argument('--continue-baseline', action='store_true',
                        help='Continue training baseline model (resume from checkpoint)')
    parser.add_argument('--train-transfer', action='store_true',
                        help='Train transfer learning model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate a trained model')
    parser.add_argument('--realtime', action='store_true',
                        help='Run real-time detection')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize model for deployment')

    # Common arguments
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=['baseline', 'mobilenetv2'],
                        help='Model to use (default: mobilenetv2)')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--phase1-epochs', type=int,
                        help='Phase 1 epochs for transfer learning')
    parser.add_argument('--phase2-epochs', type=int,
                        help='Phase 2 epochs for transfer learning')

    # Configuration
    parser.add_argument('--config', action='store_true',
                        help='Display current configuration')

    args = parser.parse_args()

    # Display banner
    print("\n" + "="*70)
    print(" " * 15 + "WASTE CLASSIFICATION SYSTEM v2.0")
    print(" " * 20 + "Production-Ready Pipeline")
    print("="*70)

    # Show configuration
    if args.config:
        print("\n📋 Current Configuration:")
        print(f"   - Image size: {IMG_SIZE}")
        print(f"   - Batch size: {BATCH_SIZE}")
        print(f"   - Number of classes: {NUM_CLASSES}")
        print(f"   - Classes: {CLASS_NAMES}")
        print(f"   - Random seed: {RANDOM_SEED}")
        print(f"   - Dataset split: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
        print(f"\n   - Data directory: {DATA_DIR}")
        print(f"   - Models directory: {MODELS_DIR}")
        print(f"   - Reports directory: {REPORTS_DIR}")
        return

    # Handle execution modes
    success = True

    if args.quick:
        # Quick mode: run all with reduced epochs
        print("\n🚀 Quick Mode: Running full pipeline with reduced epochs")
        args.epochs = 5
        args.phase1_epochs = 3
        args.phase2_epochs = 3
        if not (run_week1(args) and run_week2(args)):
            success = False

    elif args.all:
        # Run all weeks
        if not (run_week1(args) and run_week2(args) and run_week4(args)):
            success = False

    elif args.week:
        # Run specific week
        if args.week == 1:
            success = run_week1(args)
        elif args.week == 2:
            success = run_week2(args)
        elif args.week == 3:
            success = run_week3(args)
        elif args.week == 4:
            success = run_week4(args)

    # Individual tasks
    elif args.explore:
        success = run_script("scripts/01_data_exploration.py")

    elif args.preprocess:
        success = run_script("scripts/02_preprocessing.py")

    elif args.train_baseline:
        script_args = []
        if args.epochs:
            script_args.extend(['--epochs', str(args.epochs)])
        success = run_script("scripts/03_baseline_training.py", script_args)

    elif args.continue_baseline:
        script_args = []
        if args.epochs:
            script_args.extend(['--epochs', str(args.epochs)])
        success = run_script("scripts/07_continue_baseline_training.py", script_args)

    elif args.train_transfer:
        script_args = []
        if args.phase1_epochs:
            script_args.extend(['--phase1-epochs', str(args.phase1_epochs)])
        if args.phase2_epochs:
            script_args.extend(['--phase2-epochs', str(args.phase2_epochs)])
        success = run_script("scripts/04_transfer_learning.py", script_args)

    elif args.evaluate:
        success = run_script("scripts/99_evaluate_model.py", ['--model', args.model])

    elif args.realtime:
        success = run_script("scripts/05_realtime_detection.py", ['--model', args.model])

    elif args.optimize:
        success = run_script("scripts/06_model_optimization.py", ['--model', args.model])

    else:
        parser.print_help()
        return

    # Final summary
    print("\n" + "="*70)
    if success:
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("❌ PIPELINE FAILED!")
    print("="*70 + "\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
