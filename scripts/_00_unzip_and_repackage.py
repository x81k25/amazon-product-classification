# internal library imports
import gzip
import json

# third-party imports
import polars as pl

# -----------------------------------------------------------------------------
# unzip and repackage raw data
# -----------------------------------------------------------------------------

# extract files from gzip
with gzip.open(
    './data/zipped/products_labeled.json.gz',
    'rt',
    encoding='utf-8'
) as file:
    products_labeled_dict = json.load(file)

with gzip.open(
    './data/zipped/products_unlabeled.json.gz',
    'rt',
    encoding='utf-8'
) as file:
    products_unlabeled_dict = json.load(file)

# convert from raw json to dataframe
products_labeled_df = pl.from_dicts(products_labeled_dict)
products_unlabeled_df = pl.from_dicts(products_unlabeled_dict)

# save raw dataframe for use within notebooks
products_labeled_df.write_parquet('./data/00_products_labeled.parquet')
products_unlabeled_df.write_parquet('./data/00_products_unlabeled.parquet')

# -----------------------------------------------------------------------------
# end of _00_unzip_and_repackage.py
# -----------------------------------------------------------------------------
