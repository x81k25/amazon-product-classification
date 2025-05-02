# internal imports
import os
from datetime import datetime
import glob
import json
import pickle
import re

# third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, top_k_accuracy_score
)
import shap

# -----------------------------------------------------------------------------
# read in data
# -----------------------------------------------------------------------------

# read in latest best xgb model
model_files = glob.glob(os.path.join('./data/model_artifacts', 'xgb_best_model_*.pkl'))

if not model_files:
    raise FileNotFoundError("No XGBoost model files found in the directory")

# Extract datetime from filenames and find the latest
# The pattern captures YYYYMMDD and HHMMSS as groups
pattern = r'xgb_best_model_(\d{8})_(\d{6})\.pkl'

# Sort files by date and time in descending order
latest_file = max(model_files, key=lambda f:
''.join(re.findall(pattern, os.path.basename(f))[0])
if re.findall(pattern, os.path.basename(f)) else '')

# Load the model
with open(latest_file, 'rb') as f:
    xgb_model = pickle.load(f)

# read in model results
predictions_df = pd.read_parquet('./data/02_predictions.parquet')

# Extract y_test and y_pred
y_test = predictions_df['y_test'].values
y_pred = predictions_df['y_pred'].values

# read in category mappings
with open('./data/02_category_mapping.json', 'r', encoding='utf-8') as file:
    category_mapping = json.load(file)

# Extract probability columns and create y_pred_proba array
prob_cols = [col for col in predictions_df.columns if
             col.startswith('y_proba_')]
y_pred_proba = predictions_df[prob_cols].values

# Create reverse mapping (from numeric to category names)
reverse_mapping = {v: k for k, v in category_mapping.items()}

# read in test features
X_test = pd.read_parquet('./data/02_x_test.parquet')

# -----------------------------------------------------------------------------
# overall model metrics
# -----------------------------------------------------------------------------

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro')
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted')
top3_accuracy = top_k_accuracy_score(y_test, y_pred_proba, k=3)
top5_accuracy = top_k_accuracy_score(y_test, y_pred_proba, k=5)

# Create overall metrics DataFrame
overall_metrics = pd.DataFrame({
    'metric': ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
               'precision_weighted', 'recall_weighted', 'f1_weighted',
               'top3_accuracy', 'top5_accuracy'],
    'value': [accuracy, precision_macro, recall_macro, f1_macro,
              precision_weighted, recall_weighted, f1_weighted,
              top3_accuracy, top5_accuracy]
})

# -----------------------------------------------------------------------------
# per-class metrics
# -----------------------------------------------------------------------------

# Get unique class indices
class_indices = sorted(set(y_test).union(set(y_pred)))

# Get precision, recall, f1-score, and support values for each class
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

# Calculate per-class accuracy
class_metrics = []
for i, class_idx in enumerate(class_indices):
    if class_idx in reverse_mapping:
        # Class accuracy
        class_mask = (y_test == class_idx)
        class_correct = (y_pred[class_mask] == class_idx)
        class_accuracy = sum(class_correct) / sum(class_mask) if sum(
            class_mask) > 0 else 0

        # Store metrics
        class_metrics.append({
            'class_id': class_idx,
            'class_name': reverse_mapping[class_idx],
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i],
            'accuracy': class_accuracy
        })

# Create per-class metrics DataFrame and sort by support
per_class_df = pd.DataFrame(class_metrics).sort_values(by='support',
                                                       ascending=False)

# -----------------------------------------------------------------------------
# confusion matrix values
# -----------------------------------------------------------------------------
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=class_indices)

# Convert to DataFrame with class names
cm_df = pd.DataFrame(
    cm,
    index=[reverse_mapping.get(idx, f"Unknown-{idx}") for idx in class_indices],
    columns=[reverse_mapping.get(idx, f"Unknown-{idx}") for idx in
             class_indices]
)

# -----------------------------------------------------------------------------
# save perofmrance metrics
# -----------------------------------------------------------------------------

# Save metrics to CSV files
overall_metrics.to_parquet('./data/03_overall_metrics.parquet')
per_class_df.to_parquet('./data/03_per_class_metrics.parquet')
cm_df.to_parquet('./data/03_confusion_matrix.parquet')

# Print summary
print(f"Model Performance Summary:")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}")
print(f"Top-3 Accuracy: {top3_accuracy:.4f}")

print("\nTop 5 Classes by F1 Score:")
print(per_class_df.sort_values('f1_score', ascending=False)[
          ['class_name', 'f1_score', 'support']].head(5))

print("\nBottom 5 Classes by F1 Score:")
print(per_class_df.sort_values('f1_score')[
          ['class_name', 'f1_score', 'support']].head(5))

# -----------------------------------------------------------------------------
# global feature importance
# -----------------------------------------------------------------------------

# create shap explainer
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for a sample of data (using first 100 rows to limit computation)
# Adjust sample size as needed
sample_size = min(100, len(predictions_df))
sample_indices = np.random.choice(len(predictions_df), sample_size,
                                  replace=False)

# Assuming X_test is available and aligned with predictions_df
# If not, you'll need to provide the feature data
X_sample = X_test.iloc[sample_indices]
shap_values = explainer.shap_values(X_sample)


# Get mean absolute SHAP values for global feature importance
feature_importance = {}
for class_idx in range(len(category_mapping)):
    if isinstance(shap_values, list):  # Multi-output model
        if class_idx < len(shap_values):
            feature_importance[class_idx] = np.abs(shap_values[class_idx]).mean(
                0)
    else:  # Single output model
        feature_importance[class_idx] = np.abs(shap_values).mean(0)

# Create feature importance dataframe
importance_data = []
for class_idx, importance in feature_importance.items():
    if class_idx in reverse_mapping:
        class_name = reverse_mapping[class_idx]
        for i, feature_name in enumerate(X_sample.columns):
            importance_data.append({
                'class_id': class_idx,
                'class_name': class_name,
                'feature_name': feature_name,
                'importance': importance[i]
            })

feature_importance_df = pd.DataFrame(importance_data)

# Create a summarized global SHAP dataframe across all classes
global_importance = feature_importance_df.groupby('feature_name')['importance'].apply(
    lambda x: np.mean([i if isinstance(i, (int, float)) else i.mean() for i in x])
).reset_index()

global_importance = global_importance.sort_values('importance', ascending=False)

global_importance.to_parquet('./data/03_global_importance.parquet')

# -----------------------------------------------------------------------------
# end of 03_performance_and_explainability.py
# -----------------------------------------------------------------------------