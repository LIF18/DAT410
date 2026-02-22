import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_dataset(path):

    df = pd.read_csv(path)
    df = df.dropna()
    features = df.drop(columns=['PM_HIGH'])
    pm = df['PM_HIGH']

    return features, pm

class kmeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, number_clusters = 2):

        self.n_clusters = number_clusters
        self.kmeans = KMeans(n_clusters=number_clusters, random_state=42)
        self.cluster_to_label_dictionary = {}

    def training(self, x, y):

        self.kmeans.fit(x)
        clusters = self.kmeans.labels_
        y_array = np.array(y)

        for i in range(self.n_clusters):
            mask_boolean = (clusters == i)

            if np.sum(mask_boolean) > 0:
                labels_in_cluster = y_array[mask_boolean]
                # find the most frequent label
                majority_label = np.bincount(labels_in_cluster.astype(int)).argmax()
                self.cluster_to_label_dictionary[i] = majority_label
            else:
                # for empty clusters: assign global majority
                self.cluster_to_label_dictionary[i] = np.bincount(y_array.astype(int)).argmax()
        
        return self
    
    def prediction(self, x):
        clusters = self.kmeans.predict(x)
        predictions = np.array([self.cluster_to_label_dictionary[i] for i in clusters])

        return predictions
    
    def get_score(self, x, y):
        predictions = self.prediction(x)
        return accuracy_score(y, predictions)

def run_AITools():

    x_bj, y_bj = load_dataset('Cities/Beijing_labeled.csv')
    x_sy, y_sy = load_dataset('Cities/Shenyang_labeled.csv')
    x_gz, y_gz = load_dataset('Cities/Guangzhou_labeled.csv')
    x_sh, y_sh = load_dataset('Cities/Shanghai_labeled.csv')
    
    # Combine Beijing and Shenyang for training and validation
    x_train = pd.concat([x_bj, x_sy], axis=0)
    y_train = pd.concat([y_bj, y_sy], axis=0)

    print(f"Training Data Size : {x_train.shape}")

    sclaer = StandardScaler()
    sclaer.fit(x_train)

    x_train_scaled = sclaer.transform(x_train)
    x_gz_scaled = sclaer.transform(x_gz)
    x_sh_scaled = sclaer.transform(x_sh)

    # Split dataset into train and validation to find optimal k
    x_opt_train, x_opt_val, y_opt_train, y_opt_val = train_test_split(x_train_scaled, y_train, test_size=0.2, random_state=18)

    best_k = 2
    best_validation_score = 0

    for k in [2, 3, 4, 5, 8, 10, 15, 20, 25, 50, 75, 100, 200]:
        tmp_model = kmeansClassifier(number_clusters=k)
        tmp_model.training(x_opt_train, y_opt_train)

        train_acc = tmp_model.get_score(x_opt_train, y_opt_train)
        val_acc = tmp_model.get_score(x_opt_val, y_opt_val)

        print(f"k = {k}: train acc = {train_acc:.4f} val acc = {val_acc:.4f}")
        
        if val_acc > best_validation_score:
            best_validation_score = val_acc
            best_k = k

    print (f"best k : {best_k} with val acc {best_validation_score:.4f}")

    # final model
    final_model = kmeansClassifier(number_clusters=best_k)
    final_model.training(x_train_scaled, y_train)

    final_train_acc = final_model.get_score(x_train_scaled, y_train)
    gz_acc = final_model.get_score(x_gz_scaled, y_gz)
    sh_acc = final_model.get_score(x_sh_scaled, y_sh)

    print(f"final training accuarcy (Beijing + Shenyang): {final_train_acc:.4f}")
    print(f"Evaluation on Guangzhou: {gz_acc:.4f}")
    print(f"Evaluation on Shanghai:  {sh_acc:.4f}")


if __name__ == "__main__":
    try:
        run_AITools()
    except FileNotFoundError as e:
        print(f"Error: {e}")