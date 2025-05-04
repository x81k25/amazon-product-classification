# internal imports
import os
import argparse
import glob
import re
import pickle
import json

# third-party imports
import pandas as pd


def main(data_dir):
	"""
	Apply the trained model to unlabeled data and save predictions.
	"""
	print("Starting model application process...")

	# -----------------------------------------------------------------------------
	# read in data
	# -----------------------------------------------------------------------------
	print("Loading engineered unlabeled data...")

	# Load the engineered unlabeled data
	unlabeled_data = pd.read_parquet(
		os.path.join(data_dir, "01_products_engineered.parquet")
	)

	# -----------------------------------------------------------------------------
	# find and load the latest model
	# -----------------------------------------------------------------------------
	print("Finding and loading the latest model...")

	# Find all model files
	model_files = glob.glob(
		os.path.join('./data/model_artifacts', 'xgb_best_model_*.pkl')
	)

	if not model_files:
		raise FileNotFoundError("No XGBoost model files found in the directory")

	# Extract datetime from filenames and find the latest
	pattern = r'xgb_best_model_(\d{8})_(\d{6})\.pkl'

	# Sort files by date and time in descending order
	latest_file = max(model_files, key=lambda f:
	''.join(re.findall(pattern, os.path.basename(f))[0])
	if re.findall(pattern, os.path.basename(f)) else '')

	print(f"Using model file: {latest_file}")

	# Load the model
	with open(latest_file, 'rb') as f:
		model = pickle.load(f)

	# -----------------------------------------------------------------------------
	# apply model to unlabeled data
	# -----------------------------------------------------------------------------
	print("Applying model to unlabeled data...")

	# Make predictions
	y_pred = model.predict(unlabeled_data)
	y_pred_proba = model.predict_proba(unlabeled_data)

	# Create results dataframe
	results_df = pd.DataFrame()

	# Add prediction column
	results_df['y_pred'] = y_pred

	# Add probability columns
	for i in range(y_pred_proba.shape[1]):
		results_df[f'y_proba_{i}'] = y_pred_proba[:, i]

	# -----------------------------------------------------------------------------
	# load category mapping to convert numeric predictions to category names
	# -----------------------------------------------------------------------------
	try:
		with open(os.path.join(data_dir, '02_category_mapping.json'), 'r',
				  encoding='utf-8') as file:
			category_mapping = json.load(file)

		# Create reverse mapping (from numeric to category names)
		reverse_mapping = {int(v): k for k, v in category_mapping.items()}

		# Add category name column
		results_df['category'] = results_df['y_pred'].map(reverse_mapping)

		# Reorder columns to put category first
		cols = results_df.columns.tolist()
		cols.insert(0, cols.pop(cols.index('category')))
		results_df = results_df[cols]

	except (FileNotFoundError, json.JSONDecodeError) as e:
		print(f"Warning: Could not load category mapping - {e}")
		print("Results will contain numeric category indices only.")

	# -----------------------------------------------------------------------------
	# save results
	# -----------------------------------------------------------------------------
	print("Saving prediction results...")

	# Save as parquet
	results_df.to_parquet(os.path.join(data_dir, "04_applied_results.parquet"))

	# Save as CSV
	results_df.to_csv(os.path.join(data_dir, "04_applied_results.csv"),
					  index=False)

	print(f"Applied model to {len(unlabeled_data)} unlabeled records")
	print(f"Results saved to:")
	print(f" - {os.path.join(data_dir, '04_applied_results.parquet')}")
	print(f" - {os.path.join(data_dir, '04_applied_results.csv')}")
	print("Model application process completed successfully")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Apply trained model to unlabeled data")
	parser.add_argument("--data-dir", required=True,
						help="Directory containing data files")
	args = parser.parse_args()

	main(args.data_dir)

# -----------------------------------------------------------------------------
# end of _04_apply_model.py
# -----------------------------------------------------------------------------