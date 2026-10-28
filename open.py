import pandas as pd
import numpy as np
from tree import DecisionTree

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

if 'age' in train_data.columns:
    train_data['age'] = train_data['age'] / 365.25
if 'age' in test_data.columns:
    test_data['age'] = test_data['age'] / 365.25

target_column = 'cardio'
base_features = [col for col in train_data.columns if col not in [target_column, 'id']]
y = train_data[target_column].to_numpy()

np.random.seed(42)
indices = np.arange(len(train_data))
np.random.shuffle(indices)

split_idx = int(0.8 * len(train_data))
train_idxs, val_idxs = indices[:split_idx], indices[split_idx:]

best_accuracy = 0
best_config = {
    'dropped_column': None,
    'max_depth': 6,
    'min_samples_split': 10,
    'features': base_features
}

for col_to_drop in base_features:
    current_features = [f for f in base_features if f != col_to_drop]
    
    X_subset = train_data[current_features].to_numpy()
    X_tr, y_tr = X_subset[train_idxs], y[train_idxs]
    X_val, y_val = X_subset[val_idxs], y[val_idxs]
    
    for depth in range(2, 16):
        for min_samples in range(2, 21):
            temp_tree = DecisionTree(max_depth=depth, min_samples_split=min_samples)
            temp_tree.fit(X_tr, y_tr)
            
            val_predictions = temp_tree.predict(X_val)
            accuracy = np.mean(val_predictions == y_val) * 100
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = {
                    'dropped_column': col_to_drop,
                    'max_depth': depth,
                    'min_samples_split': min_samples,
                    'features': current_features
                }

X_full_best = train_data[best_config['features']].to_numpy()

final_tree = DecisionTree(
    max_depth=best_config['max_depth'], 
    min_samples_split=best_config['min_samples_split']
)
final_tree.fit(X_full_best, y)

X_test = test_data[best_config['features']].to_numpy()
predictions = final_tree.predict(X_test)

if 'id' in test_data.columns:
    submission_ids = test_data['id']
else:
    submission_ids = range(len(predictions))

submission = pd.DataFrame({
    'id': submission_ids,
    'cardio': predictions
})

submission.to_csv('my_submission.csv', index=False)