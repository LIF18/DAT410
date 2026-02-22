import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. The Classifier Class
# ==========================================
class KMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=3):
        """
        Args:
            n_clusters (int): The number of clusters (k) for K-means.
        """
        self.n_clusters = n_clusters
        # Use a fixed random_state for reproducibility
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_to_label_mapping_ = {}

    def fit(self, X, y):
        """
        Training logic:
        1. Cluster the data X.
        2. For each cluster, find the majority label in y.
        """
        # Fit K-Means to the features
        self.kmeans.fit(X)
        cluster_assignments = self.kmeans.labels_
        
        # Determine the label for each cluster centroid
        # Ensure y is a numpy array for boolean indexing
        y_array = np.array(y)
        
        for i in range(self.n_clusters):
            # Get indices of data points assigned to cluster i
            mask = (cluster_assignments == i)
            
            if np.sum(mask) > 0:
                # Get the true labels for these points
                labels_in_cluster = y_array[mask]
                # Find the most frequent label (majority vote)
                # np.bincount counts occurrences of non-negative ints
                majority_label = np.bincount(labels_in_cluster.astype(int)).argmax()
                self.cluster_to_label_mapping_[i] = majority_label
            else:
                # Fallback for empty clusters (rare): assign global majority
                self.cluster_to_label_mapping_[i] = np.bincount(y_array.astype(int)).argmax()
                
        return self

    def predict(self, X):
        """
        Prediction logic:
        1. Find nearest centroid for each point in X.
        2. Map cluster ID to class label.
        """
        # 1. Infer cluster assignment
        clusters = self.kmeans.predict(X)
        
        # 2. Map to label
        predictions = np.array([self.cluster_to_label_mapping_[c] for c in clusters])
        return predictions

    def score(self, X, y):
        """Standard accuracy score"""
        preds = self.predict(X)
        return accuracy_score(y, preds)

# ==========================================
# 2. Data Loading & Preprocessing Helper
# ==========================================
def load_and_process_city(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # "Restricted to observations with no missing data"
    df = df.dropna()
    
    # Separate Features and Target
    # Target is PM_HIGH, everything else is features
    X = df.drop(columns=['PM_HIGH'])
    y = df['PM_HIGH']
    
    return X, y

# ==========================================
# 3. Execution Pipeline
# ==========================================
def run_assignment_pipeline():
    print("--- Loading Data ---")
    # Load datasets (Ensure filenames match your unzipped files)
    X_bj, y_bj = load_and_process_city('Cities/Beijing_labeled.csv')
    X_sy, y_sy = load_and_process_city('Cities/Shenyang_labeled.csv')
    X_gz, y_gz = load_and_process_city('Cities/Guangzhou_labeled.csv')
    X_sh, y_sh = load_and_process_city('Cities/Shanghai_labeled.csv')
    
    # Combine Beijing and Shenyang for Training/Validation
    X_train_raw = pd.concat([X_bj, X_sy], axis=0)
    y_train_full = pd.concat([y_bj, y_sy], axis=0)
    
    print(f"Training Data Size (Beijing + Shenyang): {X_train_raw.shape}")
    
    # Scale the data! K-means is distance-based, scaling is mandatory.
    scaler = StandardScaler()
    # Fit scaler on training data only
    scaler.fit(X_train_raw)
    
    X_train_scaled = scaler.transform(X_train_raw)
    X_gz_scaled = scaler.transform(X_gz)
    X_sh_scaled = scaler.transform(X_sh)
    
    # Split Beijing+Shenyang into Train and Val to find optimal k
    # (Using 20% for validation)
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_train_scaled, y_train_full, test_size=0.2, random_state=42
    )
    
    print("\n--- Tuning Hyperparameter k (Clusters) ---")
    best_k = 2
    best_val_score = 0
    
    # Try k from 2 to 20 (or similar reasonable range)
    for k in [2, 3, 4, 5, 8, 10, 15, 20, 25, 50, 70, 75, 80, 85, 100]:
        clf = KMeansClassifier(n_clusters=k)
        clf.fit(X_opt_train, y_opt_train)
        
        train_acc = clf.score(X_opt_train, y_opt_train)
        val_acc = clf.score(X_opt_val, y_opt_val)
        
        print(f"k={k}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_k = k
            
    print(f"\nBest k found: {best_k} (Val Acc: {best_val_score:.4f})")
    
    # ==========================================
    # 4. Final Evaluation
    # ==========================================
    print("\n--- Final Evaluation ---")
    # Retrain on ALL training data (Beijing + Shenyang) with best k
    final_model = KMeansClassifier(n_clusters=best_k)
    final_model.fit(X_train_scaled, y_train_full)
    
    train_acc_final = final_model.score(X_train_scaled, y_train_full)
    gz_acc = final_model.score(X_gz_scaled, y_gz)
    sh_acc = final_model.score(X_sh_scaled, y_sh)
    
    print(f"Final Training Accuracy (Beijing+Shenyang): {train_acc_final:.4f}")
    print("-" * 30)
    print(f"Evaluation on Guangzhou: {gz_acc:.4f}")
    print(f"Evaluation on Shanghai:  {sh_acc:.4f}")

if __name__ == "__main__":
    try:
        run_assignment_pipeline()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV files are in the same directory as this script.")