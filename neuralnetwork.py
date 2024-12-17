import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

folder_path = "/home/axfrl/Documents/X_3A/INF554/projet_twitter/code_clean"

# === 2. Définir le réseau de neurones ===
class NeuralNetwork:
    def __init__(self):
        # Charger les fichiers CSV
        df_sana_path = folder_path+'/data/df_train2.csv'
        df_emo_kw_path = folder_path+'/data/keyword_emotion_features_trainset.csv'
        
        df_sana = pd.read_csv(df_sana_path)
        df_emo_kw = pd.read_csv(df_emo_kw_path)
        
        # Fusionner les DataFrames sur la colonne 'ID'
        df_train = pd.merge(df_sana, df_emo_kw, on='ID', how='outer')
        
        df_train = df_train.fillna(0)  # Remplacer les NaN par 0
        
        # Séparer les features (X) et les labels (y)
        X = df_train.drop(columns=['labels'])
        X = X.drop(columns=[str(k) for k in range(200)]).values
        y = df_train['labels']
        
        # Diviser en ensemble d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalisation des données
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Initialisation du modèle MLP
        self.model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,), max_iter=10000, random_state=42)

    def train(self):
        # Entraîner le modèle
        self.model.fit(self.X_train, self.y_train)
        
        # Prédiction sur l'ensemble de test
        y_pred = self.model.predict(self.X_test)

        # Calcul de l'accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

    def predict(self, X_new):
        # Prédire avec de nouvelles données
        X_new_scaled = self.scaler.transform(X_new)  # Normaliser les nouvelles données
        return self.model.predict(X_new_scaled)

    def prediction(self, path):
        self.train()
        testset_sana_path = path + '/df_test.csv'
        testset_emo_kw_path = path + '/keyword_emotion_features_testset.csv'
        
        df_test_sana = pd.read_csv(testset_sana_path)
        df_test_emo_kw = pd.read_csv(testset_emo_kw_path)
        
        df_test = pd.merge(df_test_sana, df_test_emo_kw, on='ID', how='outer')
        df_test = df_test.fillna(0)

        X_test = df_test.drop(columns=[str(k) for k in range(200)]).values
        y_pred = self.model.predict(X_test)
        
        # Construction du dataframe
        pred_df = pd.DataFrame()
        pred_df['ID']=df_test['ID']
        pred_df['EventType']= y_pred
        
        path_file = path+"/nn_predictions.csv"
        
        pred_df.to_csv(path_file, index=False)
        
        print(f"Le fichier CSV a été sauvegardé avec succès sous : {path_file}")
# Initialisation et entraînement du modèle


# Prédiction avec de nouvelles données (si nécessaire)
# new_data = np.array([...])  # Remplacez par de nouvelles données
# predictions = nn.predict(new_data)
# print(predictions)
