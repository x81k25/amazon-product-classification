# internal imports
import os
import argparse
import glob
import re
import pickle
import json

# third-party imports
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import TruncatedSVD, SparsePCA
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm	

def main(data_dir):
	"""
	Apply the trained model to unlabeled data and save predictions.
	"""
	print("Starting model application process...")

	# -----------------------------------------------------------------------------
	# read in data
	# -----------------------------------------------------------------------------
	print("Loading unlabeled data...")

	# Load the unzipped unlabeled data
	unlabeled_data = pl.read_parquet(
		os.path.join(data_dir, "00_products_unzipped.parquet")
	)

	# Load normalization constants from training
	with open(os.path.join('./data', '01_normalization_constants.json'), 'r') as f:
		norm_constants = json.load(f)
		log_min = norm_constants['log_min']
		log_max = norm_constants['log_max']

	# -----------------------------------------------------------------------------
	# find and load the latest models
	# -----------------------------------------------------------------------------
	print("Finding and loading the latest models...")

	# Find all model files by type
	model_files = glob.glob(os.path.join('./data/model_artifacts', 'xgb_best_model_*.pkl'))
	sparse_pca_files = glob.glob(os.path.join('./data/model_artifacts', 'sparse_pca_*.pkl'))
	svd_title_files = glob.glob(os.path.join('./data/model_artifacts', 'svd_title_*.pkl'))
	svd_features_files = glob.glob(os.path.join('./data/model_artifacts', 'svd_features_*.pkl'))
	svd_desc_files = glob.glob(os.path.join('./data/model_artifacts', 'svd_description_*.pkl'))

	if not all([model_files, sparse_pca_files, svd_title_files, svd_features_files, svd_desc_files]):
		raise FileNotFoundError("Missing required model artifacts")

	# Extract datetime and find latest for each type
	pattern = r'(\d{8})_(\d{6})\.pkl'

	def get_latest_file(file_list):
		return max(file_list, key=lambda f:
			''.join(re.findall(pattern, os.path.basename(f))[0])
			if re.findall(pattern, os.path.basename(f)) else '')

	latest_model_file = get_latest_file(model_files)
	latest_sparse_pca_file = get_latest_file(sparse_pca_files)
	latest_svd_title_file = get_latest_file(svd_title_files)
	latest_svd_features_file = get_latest_file(svd_features_files)
	latest_svd_desc_file = get_latest_file(svd_desc_files)

	print(f"Loading artifacts:")
	print(f" - Model: {os.path.basename(latest_model_file)}")
	print(f" - SparsePCA: {os.path.basename(latest_sparse_pca_file)}")
	print(f" - SVD transformers: {os.path.basename(latest_svd_title_file)}")

	# Load all models
	with open(latest_model_file, 'rb') as f:
		model = pickle.load(f)

	with open(latest_sparse_pca_file, 'rb') as f:
		sparse_pca = pickle.load(f)

	with open(latest_svd_title_file, 'rb') as f:
		svd_title = pickle.load(f)

	with open(latest_svd_features_file, 'rb') as f:
		svd_features = pickle.load(f)

	with open(latest_svd_desc_file, 'rb') as f:
		svd_description = pickle.load(f)

	# -----------------------------------------------------------------------------
	# feature engineering for model inference
	# -----------------------------------------------------------------------------
	print("Applying feature engineering transformations...")

	# standardize column names
	unlabeled_data = unlabeled_data.rename(
		lambda column_name: re.sub(r'[^a-zA-Z0-9]', '_', column_name.lower()))

	# -----------------------------------------------------------------------------
	# numeric fields
	# -----------------------------------------------------------------------------
	print("Processing numeric fields...")

	# normalize price and apply log transform (matching training preprocessing)
	unlabeled_data = unlabeled_data.with_columns(
		price_log=pl.when(~pl.col('price').is_null())
		.then(pl.col('price').log())
		.otherwise(pl.lit(None))
	)

	# Apply same normalization constants from training
	# Note: You'll need to save these constants during training or recalculate
	log_min = unlabeled_data.select(pl.col("price_log").min()).item()
	log_max = unlabeled_data.select(pl.col("price_log").max()).item()

	unlabeled_data = unlabeled_data.with_columns(
		pl.when(~pl.col("price_log").is_null())
		.then((pl.col("price_log") - log_min) / (log_max - log_min))
		.otherwise(pl.lit(None))
		.alias("price_log_norm")
	)

	unlabeled_data = unlabeled_data.with_columns(
		price_log_norm=pl.col('price_log_norm').cast(pl.Float32)
	)

	unlabeled_data = unlabeled_data.drop(['price', 'price_log'])

	# -----------------------------------------------------------------------------
	# details fields preprocessing and PCA application
	# -----------------------------------------------------------------------------
	print("Processing details fields...")

	details_df = unlabeled_data.select(
		pl.col("details").struct.field('*')
	).rename(lambda column_name: f"details_{column_name}")

	# Apply same preprocessing as training
	column_names = details_df.columns
	lowercase_names = [name.lower() for name in column_names]

	duplicates = {}
	for i, name in enumerate(lowercase_names):
		if lowercase_names.count(name) > 1:
			if name not in duplicates:
				duplicates[name] = []
			duplicates[name].append(column_names[i])

	for lowercase_name, dup_columns in duplicates.items():
		new_name = re.sub(r'[^a-zA-Z0-9]', '_', lowercase_name)
		details_df = details_df.with_columns(
			pl.coalesce([pl.col(col) for col in dup_columns]).alias(new_name)
		)
		details_df = details_df.drop(dup_columns)

	details_df = details_df.rename(
		lambda column_name: re.sub(r'[^a-zA-Z0-9]', '_', column_name.lower()))

	# Binary encoding
	details_cols = [col for col in details_df.columns if "details_" in col]
	for col in details_cols:
		details_df = details_df.with_columns(
			pl.when(pl.col(col).is_null())
			.then(pl.lit(0))
			.otherwise(pl.lit(1))
			.alias(col)
		)

	# Get expected feature names from the saved SparsePCA model
	expected_features = sparse_pca.n_features_in_
	current_features = len(details_cols)

	print(f"Expected features: {expected_features}, Current features: {current_features}")

	# Get the feature names from training (you'll need to save these during training)
	# For now, pad with zeros if fewer features, truncate if more
	if current_features < expected_features:
		# Add missing features as zeros
		for i in range(current_features, expected_features):
			details_df = details_df.with_columns(
				pl.Series(f"details_missing_{i}", [0] * len(details_df))
			)
	elif current_features > expected_features:
		# Keep only the first expected_features columns
		keep_cols = details_cols[:expected_features]
		details_df = details_df.select(keep_cols)

	# Update details_cols after modification
	details_cols = [col for col in details_df.columns if "details_" in col]

	# Apply saved SparsePCA transformer
	sparse_matrix = details_df.to_numpy()
	reduced_sparse = sparse_pca.transform(sparse_matrix).astype(np.float32)

	for i in range(reduced_sparse.shape[1]):
		unlabeled_data = unlabeled_data.with_columns(
			pl.Series(f"details_{i}", reduced_sparse[:, i])
		)

	unlabeled_data = unlabeled_data.drop('details')

	# -----------------------------------------------------------------------------
	# text fields preprocessing and embedding
	# -----------------------------------------------------------------------------
	print("Processing text fields...")

	def clean_for_bert(text):
		import re
		return re.sub(r'[^\w\s.,;:!?\'"-]', ' ', text).strip()

	unlabeled_data = unlabeled_data.with_columns(
		features_concat=pl.col("features").list.join(" ").map_elements(
			clean_for_bert, return_dtype=pl.Utf8),
		description_concat=pl.col("description").list.join(" ").map_elements(
			clean_for_bert, return_dtype=pl.Utf8)
	)

	# BERT embeddings
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
	bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

	def get_bert_embedding(text, max_length=512):
		inputs = tokenizer(text, return_tensors="pt", padding=True, 
						truncation=True, max_length=max_length).to(device)
		with torch.no_grad():
			outputs = bert_model(**inputs)
		return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

	def embed_dataframe_column(df, column_name, batch_size=32):
		embeddings = []
		for i in tqdm(range(0, len(df), batch_size)):
			batch = df[i:i + batch_size][column_name].to_list()
			batch_embeddings = [get_bert_embedding(text) for text in batch]
			embeddings.extend(batch_embeddings)
		return embeddings

	# Generate and apply embeddings with saved SVD transformers
	print("Generating and transforming title embeddings...")
	title_embeddings = embed_dataframe_column(unlabeled_data, "title")
	reduced_title_emb = svd_title.transform(title_embeddings)

	for i in range(len(reduced_title_emb[0])):
		unlabeled_data = unlabeled_data.with_columns(
			pl.Series(f"title_emb_{i}", [emb[i] for emb in reduced_title_emb])
		)

	print("Generating and transforming features embeddings...")
	features_embeddings = embed_dataframe_column(unlabeled_data, "features_concat")
	reduced_features_emb = svd_features.transform(features_embeddings)

	for i in range(len(reduced_features_emb[0])):
		unlabeled_data = unlabeled_data.with_columns(
			pl.Series(f"features_emb_{i}", [emb[i] for emb in reduced_features_emb])
		)

	print("Generating and transforming description embeddings...")
	description_embeddings = embed_dataframe_column(unlabeled_data, "description_concat")
	reduced_description_emb = svd_description.transform(description_embeddings)

	for i in range(len(reduced_description_emb[0])):
		unlabeled_data = unlabeled_data.with_columns(
			pl.Series(f"desc_emb_{i}", [emb[i] for emb in reduced_description_emb])
		)

	# Clean up unused fields
	unlabeled_data = unlabeled_data.drop([
		'title', 'features', 'description', 
		'features_concat', 'description_concat',
		'sku', 'manufacturer'
	])

	print("Feature engineering completed")

	# -----------------------------------------------------------------------------
	# apply model to unlabeled data
	# -----------------------------------------------------------------------------
	print("Applying model to unlabeled data...")

	# Convert to pandas for sklearn compatibility
	unlabeled_data_pd = unlabeled_data.to_pandas()

	# Make predictions
	y_pred = model.predict(unlabeled_data_pd)
	y_pred_proba = model.predict_proba(unlabeled_data_pd)

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