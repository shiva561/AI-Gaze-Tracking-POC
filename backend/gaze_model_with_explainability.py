
import pandas as pd
import numpy as np
import shap
import json
import os
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create artifacts folder
os.makedirs('artifacts', exist_ok=True)

# Load dataset
csv_path = 'Synthetic_Gaze_Dataset.csv'
df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

X = df[['iris_offset_left', 'iris_offset_right', 'iris_offset_top', 'iris_offset_bottom']]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    'Scaled Random Forest': Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))]),
    'Scaled Gradient Boosting': Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42))])
}

# Train & evaluate
results = {}
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'test_accuracy': test_accuracy}
    if test_accuracy > best_score:
        best_score = test_accuracy
        best_model = model
        best_name = name

print(f"Best Model: {best_name} with accuracy {best_score:.4f}")
joblib.dump(best_model, 'artifacts/gaze_direction_model_optimized.pkl')

# Confusion Matrix
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('artifacts/confusion_matrix.png')
plt.close()

# SHAP Explainability
try:
    if hasattr(best_model, 'predict_proba'):
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig('artifacts/shap_summary.png')
        plt.close()
        shap.plots.bar(shap_values, show=False)
        plt.savefig('artifacts/shap_bar.png')
        plt.close()
except Exception as e:
    print(f"SHAP error: {e}")

# LIME Explainability
try:
    lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=np.unique(y), discretize_continuous=True)
    exp = lime_explainer.explain_instance(X_test.iloc[0].values, best_model.predict_proba, num_features=4)
    exp.save_to_file('artifacts/lime_explanation.html')
except Exception as e:
    print(f"LIME error: {e}")

# Bias check (dummy example using random gender)
df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
X_train_b, X_test_b, y_train_b, y_test_b, gender_train, gender_test = train_test_split(X, y, df['gender'], test_size=0.2, random_state=42, stratify=y)
best_model.fit(X_train_b, y_train_b)
preds = best_model.predict(X_test_b)
bias_df = pd.DataFrame({'Gender': gender_test, 'Actual': y_test_b, 'Predicted': preds})
bias_results = bias_df.groupby(['Gender', 'Actual'])['Predicted'].value_counts().unstack(fill_value=0)
bias_results.to_csv('artifacts/fairness_results.csv')

# Save results
with open('artifacts/results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Artifacts saved in 'artifacts/' folder.")
