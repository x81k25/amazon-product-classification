# standard library imports
from datetime import datetime
import json
import os
import multiprocessing
import pickle
import argparse

# third party imports
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# -----------------------------------------------------------------------------
# model training and tuning
# -----------------------------------------------------------------------------

def main(
    data_dir = './data',
    subsample_size = 10000,
    random_iterations = 25
):
    print("Starting model training and tuning process...")

    # read in data
    input_file = os.path.join(data_dir, "01_products_engineered.parquet")
    print(f"Reading engineered data from: {input_file}")
    df = pd.read_parquet(input_file)

    # get number of threads on your machine
    num_threads = multiprocessing.cpu_count()
    print(f"Available threads: {num_threads}")

    # -----------------------------------------------------------------------------
    # label encoding and data splits
    # -----------------------------------------------------------------------------

    print("Creating label encodings and data splits...")
    # create label encoder for categories
    le = LabelEncoder()

    # Extract features and target
    X = df.drop('category', axis=1)
    y = le.fit_transform(df['category'])

    # store mapping for later reference
    category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Number of classes: {len(le.classes_)}")

    # create full data set train/test/val split
    # Extract features and target
    X = df.drop('category', axis=1)
    y = le.fit_transform(df['category'])

    # First create the full dataset splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # train/val split for full dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,  # 20% of the whole dataset
        random_state=42,
        stratify=y_train_val
    )

    # create stratified subsample train/test/val split
    print(f"Creating stratified subsample of size {subsample_size} for hyperparameter tuning...")
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=subsample_size,
        random_state=42,
        stratify=y
    )

    # Split the subsample with the same proportions
    X_train_val_sub, X_test_sub, y_train_val_sub, y_test_sub = train_test_split(
        X_sub,
        y_sub,
        test_size=0.20,
        random_state=42,
        stratify=y_sub
    )

    # train/val split for subsample
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_val_sub,
        y_train_val_sub,
        test_size=0.25,  # 20% of the subsample
        random_state=42,
        stratify=y_train_val_sub
    )

    # -----------------------------------------------------------------------------
    # create an initial model with defined hyperparameters
    # -----------------------------------------------------------------------------

    print("Training initial model with default hyperparameters...")
    # create initial xgboost model
    initial_model = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='merror',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0
    )

    initial_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Make predictions on test set
    y_pred_initial = initial_model.predict(X_test)

    # print preliminary accuracy metrics
    initial_accuracy = accuracy_score(y_pred_initial, y_test)
    print(f"Initial model accuracy: {initial_accuracy:.4f}")

    # -----------------------------------------------------------------------------
    # perform gridsearch with subsample of data to confirm hyperparams
    # -----------------------------------------------------------------------------

    print(f"Performing RandomizedSearchCV with {random_iterations} iterations...")
    # Use same param_grid as before
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    # Create base model
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='merror'
    )

    # Set up random search
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=random_iterations,
        scoring='accuracy',
        cv=5,
        n_jobs=num_threads//2,
        verbose=3,
        random_state=42  # For reproducibility
    )

    # Run random search
    print("Starting hyperparameter search...")
    random_search.fit(X_train_sub, y_train_sub)

    # best params and score
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")

    # train best model with best_params on subsample
    print("Training model with best parameters on subsample...")
    best_model = xgb.XGBClassifier(
        **random_search.best_params_,
        objective='multi:softmax',
        eval_metric='merror',
        early_stopping_rounds=15
    )

    best_model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_val_sub, y_val_sub)],
        verbose=False
    )

    # Make predictions on test set
    y_pred_random = best_model.predict(X_test_sub)

    # print preliminary accuracy metrics
    subsample_accuracy = accuracy_score(y_pred_random, y_test_sub)
    print(f"Best model accuracy on subsample: {subsample_accuracy:.4f}")

    # -----------------------------------------------------------------------------
    # build final model with full data set
    # -----------------------------------------------------------------------------

    print("Training final model with best parameters on full dataset...")
    # train best model with best_params on full data set
    full_model = xgb.XGBClassifier(
        **random_search.best_params_,
        objective='multi:softmax',
        eval_metric='merror',
        early_stopping_rounds=50
    )

    full_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Make predictions on test set
    y_pred_full = full_model.predict(X_test)
    y_pred_proba_full = full_model.predict_proba(X_test)

    # print preliminary accuracy metrics
    full_accuracy = accuracy_score(y_pred_full, y_test)
    print(f"Final model accuracy: {full_accuracy:.4f}")

    # -----------------------------------------------------------------------------
    # save results
    # -----------------------------------------------------------------------------

    print("Saving model results and artifacts...")
    # save category mapping
    category_mapping_serializable = {k: int(v) if isinstance(v, np.int64) else v for k, v in category_mapping.items()}

    with open(os.path.join(data_dir, '02_category_mapping.json'), 'w', encoding='utf-8') as file:
        json.dump(category_mapping_serializable, file)

    # save model output results
    results_df = pd.DataFrame(
        y_pred_proba_full,
        columns=[f'y_proba_{i}' for i in range(y_pred_proba_full.shape[1])]
    )

    results_df['y_pred'] = y_pred_full
    results_df['y_test'] = y_test

    results_df.to_parquet(os.path.join(data_dir, '02_predictions.parquet'))

    # save test features for shap values
    pd.DataFrame(X_test).to_parquet(os.path.join(data_dir, '02_x_test.parquet'))

    # -----------------------------------------------------------------------------
    # save model artifacts
    # -----------------------------------------------------------------------------

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a directory for saving if it doesn't exist
    save_dir = os.path.join(data_dir, "model_artifacts")
    os.makedirs(save_dir, exist_ok=True)

    # save best parameters as JSON
    params_file = os.path.join(save_dir, f"xgb_best_params_{timestamp}.json")
    with open(params_file, 'w') as f:
        json.dump(random_search.best_params_, f, indent=4)

    # save the trained model using pickle
    model_file = os.path.join(save_dir, f"xgb_best_model_{timestamp}.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(full_model, f)

    # save training metadata
    metadata = {
        "best_params": random_search.best_params_,
        "best_cv_score": float(random_search.best_score_),
        "grid_search_params": param_grid,
        "training_date": timestamp
    }

    metadata_file = os.path.join(save_dir, f"xgb_training_metadata_{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Process completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune XGBoost model for product classification")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing data files")
    parser.add_argument("--subsample-size", type=int, default=10000,
                        help="Size of stratified subsample for hyperparameter tuning")
    parser.add_argument("--random-iterations", type=int, default=25,
                        help="Number of random iterations for RandomizedSearchCV")
    args = parser.parse_args()

    main(args.data_dir, args.subsample_size, args.random_iterations)

# -----------------------------------------------------------------------------
# end of _02_model_training_and_tuning.py
# -----------------------------------------------------------------------------