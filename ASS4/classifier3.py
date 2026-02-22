import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_pickle('wdbc.pkl')
x = df.drop(columns=['id', 'malignant'])
y = df['malignant']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# max_depth=3 limits complexity to ensure interpretability.
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_clf.fit(x_train, y_train)

# Evaluate Performance
y_pred = dt_clf.predict(x_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("(Row: True, Col: Predicted | Top-Left: TN, Top-Right: FP, Bottom-Left: FN, Bottom-Right: TP)")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_clf, 
          feature_names=x.columns,  
          class_names=['Benign', 'Malignant'],
          filled=True, 
          rounded=True, 
          fontsize=12)
plt.title("Visualized Decision Rules (Interpretability)", fontsize=16)
plt.show() 
# plt.savefig('decision_tree_viz.png')