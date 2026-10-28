import pandas as pd
import numpy as np
from tree import DecisionTree  # Ensure tree.py is in the same directory

# 1. Load train and test datasets from the 'data/' folder
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

print(f"Loaded train data shape: {train_data.shape}")
print(f"Loaded test data shape: {test_data.shape}")

# 2. Define target and columns to drop
target_column = 'cardio'

train_drop_cols = ['cardio', 'alco', 'gluc', 'smoke']
if 'id' in train_data.columns:
    train_drop_cols.append('id')

# Extract features (X) and target (y)
X = train_data.drop(columns=train_drop_cols).to_numpy()
y = train_data[target_column].to_numpy()

# 3. Drop the exact same features from your test data
test_drop_cols = ['alco', 'gluc', 'smoke']
if 'id' in test_data.columns:
    test_drop_cols.append('id')

X_test = test_data.drop(columns=test_drop_cols).to_numpy()

# 4. Local Validation Split (To test hyperparameters locally without Kaggle)
print("\n--- Running Local Validation Test ---")
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

split_idx = int(0.8 * len(X))
train_idxs, val_idxs = indices[:split_idx], indices[split_idx:]

X_tr, y_tr = X[train_idxs], y[train_idxs]
X_val, y_val = X[val_idxs], y[val_idxs]

# Instantiate validation tree
val_tree = DecisionTree(max_depth=8, min_samples_split=8)
val_tree.fit(X_tr, y_tr)
val_predictions = val_tree.predict(X_val)
local_accuracy = np.mean(val_predictions == y_val) * 100
print(f"Local Validation Accuracy: {local_accuracy:.2f}%")
print("---------------------------------------\n")

# 5. Train Final Model on 100% of Training Data for Kaggle Submission
print("Training final model on full training set...")
final_tree = DecisionTree(max_depth=6, min_samples_split=10)
final_tree.fit(X, y)
print("Training complete!")

# 6. Generate predictions on test data
print("Generating predictions for test.csv...")
predictions = final_tree.predict(X_test)

# 7. Save your predictions to a CSV file formatted for Kaggle
if 'id' in test_data.columns:
    submission_ids = test_data['id']
else:
    submission_ids = range(len(predictions))

submission = pd.DataFrame({
    'id': submission_ids,
    'cardio': predictions
})

submission.to_csv('my_submission.csv', index=False)
print("Saved predictions to my_submission.csv with 'cardio' column!")