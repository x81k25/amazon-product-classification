# internal library imports
import re

# third-party imports
import numpy as np
import polars as pl
import torch
from sklearn.decomposition import TruncatedSVD, SparsePCA
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# read in labeled data
# -----------------------------------------------------------------------------

df = pl.read_parquet('./data/00_products_labeled.parquet')

# -----------------------------------------------------------------------------
# initial transformations
# -----------------------------------------------------------------------------

# standardize column names
df = df.rename(lambda column_name: re.sub(r'[^a-zA-Z0-9]', '_', column_name.lower()))

# -----------------------------------------------------------------------------
# numeric fields
# -----------------------------------------------------------------------------

# normalize amount and apply log transform
df = df.with_columns(
    price_log = pl.when(~pl.col('price').is_null())
        .then(pl.col('price').log())
        .otherwise(pl.lit(None))
)

log_min = df.select(pl.col("price_log").min()).item()
log_max = df.select(pl.col("price_log").max()).item()

# Apply min-max normalization while preserving nulls
df = df.with_columns(
    pl.when(~pl.col("price_log").is_null())
      .then((pl.col("price_log") - log_min) / (log_max - log_min))
      .otherwise(pl.lit(None))
      .alias("price_log_norm")
)

# cut precision
df = df.with_columns(
    price_log_norm = pl.col('price_log_norm').cast(pl.Float32)
)

# remove no longer used columns
df = df.drop(['price', 'price_log'])

# -----------------------------------------------------------------------------
# details fields
# -----------------------------------------------------------------------------

details_df = df.select(
    pl.col("details").struct.field('*')
).rename(lambda column_name: f"details_{column_name}")

# standardize column names and remove duplicates
column_names = details_df.columns
lowercase_names = [name.lower() for name in column_names]

duplicates = {}

# find duplicates
for i, name in enumerate(lowercase_names):
    if lowercase_names.count(name) > 1:
        if name not in duplicates:
            duplicates[name] = []
        duplicates[name].append(column_names[i])

# merge duplicate columns
for lowercase_name, dup_columns in duplicates.items():
    new_name = re.sub(r'[^a-zA-Z0-9]', '_', lowercase_name)

    # Create a coalesce expression that takes the first non-null value from duplicate columns
    details_df = details_df.with_columns(
        pl.coalesce([pl.col(col) for col in dup_columns]).alias(new_name)
    )

    # Optionally drop the original duplicate columns
    details_df = details_df.drop(dup_columns)

# standardize column names
details_df = details_df.rename(lambda column_name: re.sub(r'[^a-zA-Z0-9]', '_', column_name.lower()))

# convert all details columns to a binary encoding
details_cols = [col for col in details_df.columns if "details_" in col]

# Convert each column to integer (1 for non-null, 0 for null values)
for col in details_cols:
    details_df = details_df.with_columns(
        pl.when(pl.col(col).is_null())
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias(col)
    )

# Assuming df_sparse is your 172 x 42429 polars dataframe
sparse_matrix = details_df.to_numpy()

# Apply Sparse PCA
n_components = 50
sparse_pca = SparsePCA(
    n_components=n_components,
    alpha=1,  # Sparsity control
    random_state=42,
    n_jobs=-1  # Use all cores
)

# Transform the data
reduced_sparse = sparse_pca.fit_transform(sparse_matrix).astype(np.float32)

# add to df
for i in range(reduced_sparse.shape[1]):
    df = df.with_columns(
        pl.Series(f"details_{i}", reduced_sparse[:, i])
    )

# remove unused fields
df = df.drop('details')

# -----------------------------------------------------------------------------
# string fields
# -----------------------------------------------------------------------------

# concatenate list fields and remove special characters
def clean_for_bert(text):
    # Remove special brackets and other non-semantic characters
    # Keep alphanumeric, basic punctuation and spaces
    import re
    return re.sub(r'[^\w\s.,;:!?\'"-]', ' ', text).strip()

df = df.with_columns(
    features_concat = pl.col("features").list.join(" ").map_elements(clean_for_bert, return_dtype=pl.Utf8),
    description_concat = pl.col("description").list.join(" ").map_elements(clean_for_bert, return_dtype=pl.Utf8)
)

# get bert embeddings
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

# Function to create embeddings
def get_bert_embedding(text, max_length=512):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use the [CLS] token embedding as the sentence embedding
    # (alternatively, you could use mean pooling)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding


# Process DataFrame in batches
def embed_dataframe_column(df, column_name, batch_size=32):
    embeddings = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[i:i + batch_size][column_name].to_list()
        batch_embeddings = [get_bert_embedding(text) for text in batch]
        embeddings.extend(batch_embeddings)

    return embeddings


# Create embeddings for each column
title_embeddings = embed_dataframe_column(df, "title")

# initialize TruncatedSVD model
svd = TruncatedSVD(n_components=50, random_state=42)

# fit and transform the embeddings
reduced_title_emb = svd.fit_transform(title_embeddings)

# add embeddings to the dataframe
for i in range(len(reduced_title_emb[0])):
    df = df.with_columns(
        pl.Series(f"title_emb_{i}", [emb[i] for emb in reduced_title_emb])
    )

# repeat for features
features_embeddings = embed_dataframe_column(df, "features_concat")

reduced_features_emb = svd.fit_transform(features_embeddings)

for i in range(len(reduced_features_emb[0])):
    df = df.with_columns(
        pl.Series(f"features_emb_{i}", [emb[i] for emb in reduced_features_emb])
    )

# repeat for description
description_embeddings = embed_dataframe_column(df, "description_concat")

reduced_description_emb = svd.fit_transform(description_embeddings)

for i in range(len(reduced_description_emb[0])):
    df = df.with_columns(
        pl.Series(f"desc_emb_{i}", [emb[i] for emb in reduced_description_emb])
    )

# remove unused fields
df = df.drop(
    'title',
    'features',
    'description',
    'features_concat',
    'description_concat'
)

# -----------------------------------------------------------------------------
# prep for model_training
# -----------------------------------------------------------------------------

# drop fields that won't be used
df = df.drop(
    'sku',
    'manufacturer',
)

# -----------------------------------------------------------------------------
# write processed data
# -----------------------------------------------------------------------------

df.write_parquet("./data/01_products_engineered.parquet")

# -----------------------------------------------------------------------------
# end of _01_feature_engineering.py
# -----------------------------------------------------------------------------
