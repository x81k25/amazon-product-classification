# amazon product classification

The purpose of this repo is to take semi-structured data describing products, perform a multi-class classification on that data, and to document the problem, assumptions, solutions, and further questions.   

## project structure

```
|-- data/               # data files containing data at varisous stages of processing
|-- notebooks/          # analysis and visualization of the process 
|-- src/                # python scripts that execute all core tasks
|-- .gitignore
|-- main.py
|-- readme.md
|-- requriements.in     # for uv package install
|-- requirements.out    # for uv package install

```

## setup

1. Clone the repository:
   
```bash
git clone <repository-url>
cd <repository-name>
```
 
2. create virtual environment

python -m venv venv

3. activate the virtual environment

Windows: `venv\Scripts\activate`
macOS/Linux: `source venv/bin/activate`

4. install uv to venv

```bash
pip install uv
```

5. modify dependancies if needed via the requirements.in file

6. if step 5 needed, regenerate requirements.out

```bash
uv pip compile requirements.in -o requirements.out
```

7. install either the original or your updated requirements

```bash
uv pip install -r requirements.out
```

8. if you intend to use GPU with torch, you will also need this statement
- this command may need to be altered based on your CUDA verison
- note: the package is almost 3GB in size

```bash
uv pip install torch torchvision torchaudio --torch-backend=cu118
```

## workflow

All data required for the notebooks is contained with the data folder. They may be open, ran, and altered at any point. 

The scripts file names are indexed in sequential order. If you wish to use a new or altered version of the dataset you will need to run the scripts sequentially in order to properly generate the data for subsequent steps. The scripts include:

`00_unzip_and_repackage` 
- unzips the raw .gz data files in `./data/zipped/`
- stores raw json data as dict and converts top level to data frame
- stores labeled data as `./data/00_products_lableled.parquet`
- stores unlabeled data as `./data/00_products_unlableled.parquet`
- note: unzipping and storing the raw data as .json will create a file that take about 4x as much storage and will put you over the GitHub LFS limis 

`01_feature_engineering`
- reads in the unzipped data
- performs all feature engineering on the raw data set
- stores feature engineered data as `./data/01_products_engineered.parquet`

`02_model_training_and_tuning`
- reads in engineered data
- creates model object
- performs hyperparameter gridsearch 
- stores model artifacts to `./data/model_artifacts/`
- stores model predictions and probabilities to `./data/02_predictions.parquet`

`03_performance_and_explainabilty` 
- reads in `./data/02_predictions.parquet`
- creates model metric objects to be used by the `analysis_of_results.ipynb` notebook 
- creates explainability objects to be used by the `analysis_of_results.ipynb` notebook

## methodology

## Methodology

### Dataset
The dataset consists of 43,000 product records spanning 28 different categories. Each product record contains various fields including:
- Category (target variable)
- Price (numerical, with 58.58% null values)
- Title (text, 99.84% distinct values)
- Features (list of text strings, 83.25% distinct values)
- Description (list of text strings, 68.69% distinct values)
- Details (nested structure with product-specific attributes)

The data exploration revealed several important characteristics:
- Nearly 60% of products have missing price information
- Price values are highly skewed, ranging from $0.01 to $18,999 with a mean of $60.40
- The Details field contains over 100 different attributes with varying presence across products
- Common attributes in Details include Item model number (76.03%), Manufacturer (59.31%), and Product Dimensions (46.25%)
- Most Details attributes have high null rates, with only 20 attributes present in more than 10% of products

The data is highly heterogeneous, with different products having different available fields and structure, particularly in the Details section where the attributes vary significantly by product type.

### Problem
The problem is to develop a multi-class classifier that can accurately categorize products into one of 28 predefined categories based on their attributes. This is a supervised machine learning task where we need to predict the 'Category' field using the other available information about each product.

### Assumptions
1. Text fields (Title, Features, Description) contain valuable semantic information that can help identify product categories
2. Missing values, particularly in the Price field, don't necessarily indicate data quality issues but reflect the reality of the dataset
3. The Details field, despite its variability across products, contains useful signals for categorization when properly encoded
4. The dataset is representative of the distribution of products we expect to classify in production
5. A tree-based model would be effective for this heterogeneous data with both numerical and categorical features
6. The computational constraints require balancing model complexity with training time

### Solution

#### Data Preprocessing and Feature Engineering
The preprocessing and feature engineering pipeline was designed to handle the heterogeneous nature of the data:

1. **Numeric fields (Price):**
   - Applied log transformation to normalize the distribution
   - Performed min-max normalization while preserving null values
   - Cast to Float32 to reduce memory usage

2. **Structured fields (Details):**
   - Standardized field names using lowercase and replacing special characters
   - Identified and merged duplicate columns
   - Created binary encodings (0/1) indicating presence/absence of each attribute
   - Applied Sparse PCA to reduce dimensionality from the large sparse matrix (42,429 columns) to 50 components, preserving the most important signals while drastically reducing dimensionality

3. **Text fields (Title, Features, Description):**
   - Cleaned and preprocessed text by removing special characters
   - Concatenated list fields into single strings
   - Generated embeddings using DistilBERT to capture semantic meaning
   - Applied TruncatedSVD to reduce embedding dimensionality from 768 to 50 components for each text field, preserving computational efficiency while maintaining semantic information

This preprocessing approach resulted in a feature set that captured the essential characteristics of each product while being computationally manageable.

#### Model Selection
XGBoost was selected as the classification algorithm for several reasons:

1. **Effectiveness with heterogeneous data:** XGBoost works well with mixed data types and features of different scales
2. **Handling of missing values:** Native support for missing values, which were present in our dataset
3. **Speed and scalability:** Relatively fast training compared to deep learning approaches, meeting the time constraint requirements
4. **Strong performance on tabular data:** Consistently strong performance on structured data problems
5. **Robust to overfitting:** Regularization parameters help prevent overfitting on high-dimensional data

#### Model Training & Hyperparameter Tuning
The training process followed a systematic approach:

1. **Data splitting:**
   - Created stratified train/validation/test splits (60%/20%/20%) to ensure proper representation of all 28 categories
   - Additionally created a 10,000-sample stratified subset for efficient hyperparameter tuning

2. **Initial model:**
   - Trained an initial XGBoost model with default hyperparameters to establish a baseline
   - Used multi:softmax objective function appropriate for multi-class classification

3. **Hyperparameter optimization:**
   - Performed RandomizedSearchCV on the 10,000-sample subset to efficiently explore the hyperparameter space
   - Explored key parameters including learning rate, tree depth, number of estimators, and regularization parameters
   - Used 5-fold cross-validation to ensure robust parameter selection
   - Limited to 25 iterations to balance exploration with computational constraints

4. **Final model:**
   - Trained the final model on the full dataset using the best hyperparameters from the search
   - Implemented early stopping (50 rounds) to prevent overfitting
   - Achieved ~90-91% accuracy on the hold-out test set

5. **Model artifacts:**
   - Saved the trained model, best parameters, and training metadata for reproducibility and deployment

### Potential Alternate Solutions
Several alternative approaches could be considered:

1. **Deep Learning:** A neural network approach, particularly a multi-modal architecture that handles text and structured data separately before combining them, might achieve higher accuracy but would require significantly more training time

2. **Ensemble methods:** Combining multiple models (e.g., XGBoost, Random Forest, and Neural Networks) might improve accuracy through diverse learning approaches

3. **More extensive text preprocessing:** Advanced NLP techniques like entity recognition or topic modeling might extract more nuanced features from text fields

4. **Feature selection:** A more rigorous feature selection process could potentially identify the most predictive features and simplify the model

These alternatives were not pursued due to time constraints and the already strong performance of the XGBoost model.

### Open Questions
1. How would the model perform on new products from categories not represented in the training data?

2. Could a more sophisticated approach to handling the Details field further improve performance?

3. Would incorporating image data (if available) significantly improve classification accuracy?

4. How stable is the model performance over time as product descriptions and attributes evolve?

5. Could a simpler model achieve comparable performance with faster inference time?

6. What is the minimum set of features needed to maintain the current level of accuracy?

## license

MIT License
