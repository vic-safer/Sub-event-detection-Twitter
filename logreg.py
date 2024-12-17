from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from .preprocess import Preprocess

folder_path = "/home/axfrl/Documents/X_3A/INF554/projet_twitter/code_clean"

class LogisticRegressionClassifier:
    def __init__(self):
        self.preprocess = Preprocess()
        path = folder_path+"/data/df_by_period_nuit.csv"
        df_loss = pd.read_csv(path).fillna(0)
        self.df_train = df_loss
        self.X = self.df_train.drop(columns=['EventType', 'ID']).values
        self.y = self.df_train['EventType']
        
        self.scaler = StandardScaler()
        
        self.rf_classifier = LogisticRegression(max_iter=10000, warm_start=True, random_state=42, solver='lbfgs')
        
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # We set up a basic classifier that we train and then calculate the accuracy on our test set
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) 
        
        self.rf_classifier.fit(X_train_scaled, y_train)
        y_pred = self.rf_classifier.predict(X_test_scaled)
        print("Accuracy: ", accuracy_score(y_test, y_pred))
    
    def prediction(self, path):
        # Appel de la méthode train pour entraîner le modèle
        self.train()

        # Chargement des fichiers de test
        df_test = self.preprocess.get_preprocessed_test_input(path)

        # Standardisation des données de test
        X_test_scaled = self.scaler.transform(df_test.drop(columns=['ID']).values)

        y_pred = self.rf_classifier.predict(X_test_scaled)
        
        # Construction du dataframe
        pred_df = pd.DataFrame()
        pred_df['ID']=df_test['ID']
        pred_df['EventType']= y_pred
        
        path_file = path+"/rf_predictions.csv"
        
        pred_df.to_csv(path_file, index=False)
        
        print(f"Le fichier CSV a été sauvegardé avec succès sous : {path_file}")
