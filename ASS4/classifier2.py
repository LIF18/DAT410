import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_pickle('wdbc.pkl')

x = df.drop(columns=['id', 'malignant'])
y = df['malignant']

# Split data into training set (80%) and testing set (20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# n_estimators=100 means building 100 decision trees
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(x_train, y_train)

# Predict and Evaluate
y_pred = rf_clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("(Row: True, Col: Predicted | Top-Left: TN, Top-Right: FP, Bottom-Left: FN, Bottom-Right: TP)")
print("\nClassification Report:")
print(class_report)

# Feature Importance, very useful for model interpretability
importances = rf_clf.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

print("\nTop 5 Most Important Features:")
for i in range(5):
    idx = indices[i]
    print(f"{x.columns[idx]}: {importances[idx]:.4f}")