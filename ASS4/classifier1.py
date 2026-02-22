import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle('wdbc.pkl')

# features: area_0, concavity_0, texture_0, compactness_0
# label: malignant (1 = Malignant, 0 = Benign)
features = {
    'area_0': 'Cell Size (Area)',
    'concavity_0': 'Cell Shape (Concavity)',
    'texture_0': 'Cell Texture',
    'compactness_0': 'Cell Homogeneity (Compactness)'
}

# Calculate statistics to help judgment
stats = df.groupby('malignant')[list(features.keys())].describe(percentiles=[0.5, 0.75, 0.9, 0.95])
print(stats.T)

# Plot histograms to visually find the separation point
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Recommended thresholds based on previous analysis
recommended_thresholds = {
    'area_0': 700, 
    'concavity_0': 0.10, 
    'texture_0': 23, 
    'compactness_0': 0.12
}

for i, (col, name) in enumerate(features.items()): 
    sns.histplot(data=df, x=col, hue='malignant', kde=True, element="step", ax=axes[i], palette='Set2')
    axes[i].set_title(f'Distribution of {name}')
    axes[i].set_xlabel(col)

    threshold = recommended_thresholds[col]
    axes[i].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    axes[i].legend()

def rule_based_classifier(row, t_area=recommended_thresholds['area_0'], t_concavity=recommended_thresholds['concavity_0'], 
                          t_texture=recommended_thresholds['texture_0'], t_compactness=recommended_thresholds['compactness_0']):

    if row['area_0'] > t_area:
        return 1
    
    if row['concavity_0'] > t_concavity:
        return 1
    
    if row['texture_0'] > t_texture:
        return 1
        
    if row['compactness_0'] > t_compactness:
        return 1
        
    return 0

predicted_labels = df.apply(rule_based_classifier, axis=1)

# Evaluate Performance
y_true = df['malignant']
y_pred = predicted_labels

accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("(Row: True, Col: Predicted | Top-Left: TN, Top-Right: FP, Bottom-Left: FN, Bottom-Right: TP)")
plt.tight_layout()
plt.show()