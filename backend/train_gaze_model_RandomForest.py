import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load your dataset
df = pd.read_csv('Synthetic_Gaze_Dataset.csv')  # Ensure this is in the same folder
print(df['label'].value_counts())

# Step 2: Prepare features and labels
X = df[['iris_offset_left', 'iris_offset_right', 'iris_offset_top', 'iris_offset_bottom']]
y = df['label']

sns.pairplot(df, hue="label")
plt.show()

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Save the model to disk
joblib.dump(clf, 'gaze_direction_model.pkl')

print("✅ Model trained and saved as gaze_direction_model.pkl")
