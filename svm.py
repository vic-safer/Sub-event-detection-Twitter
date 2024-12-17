import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC  # Importation du modèle SVM
from .preprocess import Preprocess

folder_path = "/home/axfrl/Documents/X_3A/INF554/projet_twitter/code_clean"

class SVMModel:
    def __init__(self):
        # Chemins des fichiers
        df_sana_path = '/data/df_train2.csv'
        df_emo_kw_path = '/data/keyword_emotion_features_trainset.csv'
        
        # Chargement des datasets
        df_sana = pd.read_csv(df_sana_path)
        df_emo_kw = pd.read_csv(df_emo_kw_path)
        
        # Fusion des datasets
        df_train = pd.merge(df_sana, df_emo_kw, on='ID', how='outer')
        self.df_train = df_train.fillna(0)

        # Préparation des features et des labels
        self.X = self.df_train.drop(columns=['labels']).values
        self.y = self.df_train['labels'].values

        self.preprocess = Preprocess()

        self.df_train = self.preprocess.get_preprocessed_input()
        self.X = self.df_train.drop(columns=['labels', 'ID']).values
        self.y = self.df_train['labels']

        self.scaler = StandardScaler()

        # Initialisation du modèle SVM
        self.svm_classifier = SVC(kernel='rbf',coef0=0.5, gamma='scale', C=5.0, random_state=42)  # Modèle avec kernel RBF par défaut

    def display(self, path):
        print(self.df_train)
        self.df_train.to_csv(path + "/display_train.csv", index=False)
        
        # Chargement des fichiers de test
        testset_sana_path = path + '/df_test.csv'
        testset_emo_kw_path = path + '/keyword_emotion_features_testset.csv'
        
        df_test_sana = pd.read_csv(testset_sana_path)
        df_test_emo_kw = pd.read_csv(testset_emo_kw_path)
        
        # Fusion des datasets de test
        df_test = pd.merge(df_test_sana, df_test_emo_kw, on='ID', how='outer')
        df_test = df_test.fillna(0)
        
        print(df_test)
        df_test.to_csv(path + "/display_test.csv", index=False)

    def train(self):
        # Division du dataset en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Standardisation des données        
        X_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) 

        # Entraînement du modèle SVM
        self.svm_classifier.fit(X_scaled, y_train)

        # Prédictions sur les données de test
        y_pred = self.svm_classifier.predict(X_test_scaled)

        # Calcul de l'accuracy
        print("Accuracy: ", accuracy_score(y_test, y_pred))

    def prediction(self, path):
        # Appel de la méthode train pour entraîner le modèle
        self.train()

        # Chargement des fichiers de test
        df_test = self.preprocess.get_preprocessed_test_input(path)

        # Standardisation des données de test
        X_test_scaled = self.scaler.transform(df_test.drop(columns=['ID']).values)


        # Prédictions sur les données de test
        y_pred = self.svm_classifier.predict(X_test_scaled)

        # Sauvegarde des prédictions dans un fichier CSV
        pred_df = pd.DataFrame()
        pred_df['ID'] = df_test['ID']
        pred_df['EventType'] = y_pred

        path_file = path + "/svm_predictions.csv"
        pred_df.to_csv(path_file, index=False)

        print(f"Le fichier CSV a été sauvegardé avec succès sous : {path_file}")
