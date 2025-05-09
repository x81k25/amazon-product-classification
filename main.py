"""
Main entry point for Amazon product classification pipeline.
"""
import argparse
import subprocess
import sys

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Amazon Product Classification Pipeline"
    )

    # Arguments matching the 00 script with defaults
    parser.add_argument(
        "--input-zip",
        type=str,
        required=False,
        default="./data/zipped/products_labeled.json.gz",
        help="Directory containing zipped files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        default="./data",
        help="Directory to save unzipped data"
    )
    parser.add_argument(
        "--subsample-size",
        type=int,
        default=5000,
        help="Size of stratified subsample for hyperparameter tuning"
    )
    parser.add_argument(
        "--random-iterations",
        type=int,
        default=5,
        help="Number of random iterations for RandomizedSearchCV"
    )
    parser.add_argument(
        "--run",
        type=str,
        nargs="+",  # Accept multiple values
        choices=["unzip", "engineer", "train", "explain", "apply"],
        default=["unzip", "engineer", "train", "explain"],  # Default to run all steps
        help="Specify which subprocesses to run: unzip, engineer, train, explain"
    )
    args = parser.parse_args()

    # Run subprocesses based on provided arguments
    if "unzip" in args.run:
        # Run the unzip script
        cmd = [
            sys.executable,  # Use the current Python interpreter
            "scripts/_00_unzip_and_repackage.py",
            "--input-zip", args.input_zip,
            "--output-dir", args.data_dir
        ]
        subprocess.run(cmd, check=True)

    if "engineer" in args.run:
        # Run feature engineering
        cmd = [
            sys.executable,
            "scripts/_01_feature_engineering.py",
            "--input-dir", args.data_dir,
        ]
        subprocess.run(cmd, check=True)

    if "train" in args.run:
        # Run the model training script
        subprocess.run([
            sys.executable,
            "./scripts/_02_model_training_and_tuning.py",
            "--data-dir", args.data_dir,
            "--subsample-size", str(args.subsample_size),
            "--random-iterations", str(args.random_iterations)
        ])

    if "explain" in args.run:
        # Run the model explanation script (assuming you have one)
        subprocess.run([
            sys.executable,
            "./scripts/_03_performance_and_explainability.py",
            "--data-dir", args.data_dir
        ])

    if "apply" in args.run:
        # Run the model explanation script (assuming you have one)
        subprocess.run([
            sys.executable,
            "./scripts/_04_apply_model.py",
            "--data-dir", args.data_dir
        ])


if __name__ == "__main__":
    main()