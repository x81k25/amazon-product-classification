# internal library imports
import argparse
import gzip
import json
import os

# third-party imports
import polars as pl


# -----------------------------------------------------------------------------
# unzip and repackage raw data
# -----------------------------------------------------------------------------

def main(input_zip, output_dir):
	print("Starting unzip and repackage process...")

	# extract files from gzip
	with gzip.open(
		os.path.join(input_zip),
		'rt',
		encoding='utf-8'
	) as file:
		products_labeled_dict = json.load(file)

	# convert from raw json to dataframe
	products_labeled_df = pl.from_dicts(products_labeled_dict)

	# save raw dataframe for use within notebooks
	products_labeled_df.write_parquet(
		os.path.join(output_dir, '00_products_unzipped.parquet'))

	print("Process completed successfully")


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Unzip and repackage raw data")
	parser.add_argument("--input-zip", required=True,
						help="Zipped file containing a json file")
	parser.add_argument("--output-dir", required=True,
						help="Directory to save unzipped data")
	args = parser.parse_args()

	main(args.input_zip, args.output_dir)

# -----------------------------------------------------------------------------
# end of *00*unzip_and_repackage.py
# -----------------------------------------------------------------------------