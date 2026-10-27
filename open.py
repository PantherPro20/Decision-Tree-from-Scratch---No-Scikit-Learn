import pandas as pd
import numpy as np

from tree import DecisionTree

# 1. Load your training and test CSV files
# (Make sure to replace 'train.csv' and 'test.csv' with your actual file names)
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# 2. Separate features and target in your training data
# Replace 'target_column_name' with the actual name of the column you are trying to predict (e.g., 'cardio')
target_column = 'cardio' 

X_train = train_data.drop(columns=[target_column]).to_numpy()
y_train = train_data[target_column].to_numpy()

# For your test data, drop the target column if it's there, or keep just the features
# (If your test.csv doesn't have the target column, you don't need to drop it)
X_test = test_data.to_numpy()

# 3. Instantiate your custom Decision Tree
# You can tweak max_depth and min_samples_split right here on your branch!
my_tree = DecisionTree(max_depth=5, min_samples_split=10)

print("Training your custom Decision Tree...")
my_tree.fit(X_train, y_train)
print("Training complete!")

# 4. Generate predictions on your test data
predictions = my_tree.predict(X_test)

# 5. Save your predictions to a CSV file for Kaggle submission
submission = pd.DataFrame({
    'Id': range(len(predictions)), # Or use an actual ID column from your test set if available
    'Prediction': predictions
})
submission.to_csv('my_submission.csv', index=False)
print("Saved predictions to my_submission.csv!")