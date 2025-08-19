import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("🔍 Loading and analyzing dataset...")

# Load dataset
df = pd.read_csv('Synthetic_Gaze_Dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Prepare features and labels
X = df[['iris_offset_left', 'iris_offset_right', 'iris_offset_top', 'iris_offset_bottom']]
y = df['label']

# Analyze feature patterns for each direction
print("\n📊 Feature Analysis by Direction:")
for label in df['label'].unique():
    subset = df[df['label'] == label]
    print(f"\n{label.upper()}:")
    print(f"  Left:   {subset['iris_offset_left'].mean():.3f} ± {subset['iris_offset_left'].std():.3f}")
    print(f"  Right:  {subset['iris_offset_right'].mean():.3f} ± {subset['iris_offset_right'].std():.3f}")
    print(f"  Top:    {subset['iris_offset_top'].mean():.3f} ± {subset['iris_offset_top'].std():.3f}")
    print(f"  Bottom: {subset['iris_offset_bottom'].mean():.3f} ± {subset['iris_offset_bottom'].std():.3f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📈 Training Models...")

# Define models to compare
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    'Scaled Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ]),
    'Scaled Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42))
    ])
}

# Evaluate all models
results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Fit and test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'model': model
    }
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    if test_accuracy > best_score:
        best_score = test_accuracy
        best_model = model
        best_name = name

print(f"\n🏆 BEST MODEL: {best_name} (Test Accuracy: {best_score:.4f})")

# Detailed evaluation of best model
print(f"\n📋 Detailed Evaluation of {best_name}:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    features = X.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)), importances[indices])
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    plt.title(f'Feature Importance - {best_name}')
    plt.tight_layout()
    plt.show()
    
    print(f"\n🎯 Feature Importance:")
    for i in indices:
        print(f"  {features[i]}: {importances[i]:.4f}")

elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps.get('rf') or best_model.named_steps.get('gb'), 'feature_importances_'):
    # For pipeline models
    if 'rf' in best_model.named_steps:
        importances = best_model.named_steps['rf'].feature_importances_
    else:
        importances = best_model.named_steps['gb'].feature_importances_
    
    features = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)), importances[indices])
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    plt.title(f'Feature Importance - {best_name}')
    plt.tight_layout()
    plt.show()

# Save the best model
model_filename = 'gaze_direction_model_optimized.pkl'
joblib.dump(best_model, model_filename)
print(f"\n💾 Best model saved as: {model_filename}")

# Test some specific cases to understand the model behavior
print(f"\n🧪 Testing Model Predictions:")

# Test cases based on dataset patterns
test_cases = [
    ([0.1, 0.8, 0.4, 0.4], "Should be LEFT"),
    ([0.8, 0.1, 0.4, 0.4], "Should be RIGHT"), 
    ([0.4, 0.4, 0.1, 0.8], "Should be UP"),
    ([0.4, 0.4, 0.8, 0.1], "Should be DOWN"),
    ([0.45, 0.55, 0.45, 0.55], "Should be CENTER")
]

for features, expected in test_cases:
    prediction = best_model.predict([features])[0]
    print(f"  {features} → {prediction} ({expected})")

# Performance comparison chart
plt.figure(figsize=(12, 8))
model_names = list(results.keys())
cv_means = [results[name]['cv_mean'] for name in model_names]
test_accs = [results[name]['test_accuracy'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, cv_means, width, label='CV Accuracy', alpha=0.8)
plt.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n✅ Model comparison complete!")
print(f"🎯 Use the saved model: {model_filename}")
print(f"🏆 Best performing model: {best_name} with {best_score:.4f} accuracy")

# Create an enhanced feature extraction function for real implementation
print(f"\n💡 IMPORTANT: The current HTML uses simulated features.")
print(f"To get accurate results, you need to implement real eye tracking:")
print(f"1. Use MediaPipe Face Mesh for face detection")
print(f"2. Extract actual iris positions relative to eye corners")
print(f"3. Calculate real iris_offset values")
print(f"4. Replace the extractIrisFeatures() function in HTML")