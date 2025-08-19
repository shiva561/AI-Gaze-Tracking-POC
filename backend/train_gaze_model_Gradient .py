import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load dataset
df = pd.read_csv('Synthetic_Gaze_Dataset.csv')
print(df['label'].value_counts())

# Step 2: Prepare features and labels
X = df[['iris_offset_left', 'iris_offset_right', 'iris_offset_top', 'iris_offset_bottom']]
y = df['label']

# Optional: Visualize
sns.pairplot(df, hue="label")
plt.show()

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create pipeline with scaler + classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42))
])

# Step 5: Train model
pipeline.fit(X_train, y_train)

# Step 6: Save model
joblib.dump(pipeline, 'gaze_direction_model.pkl')
print("✅ Enhanced model trained and saved as gaze_direction_model.pkl")
