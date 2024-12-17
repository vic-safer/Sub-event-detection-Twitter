from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import pandas as pd
from .preprocess import Preprocess
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
import numpy as np

folder_path = "/home/axfrl/Documents/X_3A/INF554/projet_twitter/code_clean"

class XGBoostModel:
    def __init__(self, dim_pca):
        self.preprocess = Preprocess(dim_pca)
        path = folder_path+'/data/df_train_nuit_finale.csv'
        df_loss = pd.read_csv(path).fillna(0)
        self.df_train = self.preprocess.get_preprocessed_input()
        
        self.X = self.df_train.drop(columns=['EventType', 'ID']).values
        self.y = self.df_train['EventType']
        
        self.scaler = StandardScaler()
        self.params = {
                "objective": "binary:logistic",  # Si c'est un problème de classification binaire
                "eval_metric": "auc",       # Fonction de perte
                "learning_rate": 0.04,          # Faible taux d'apprentissage
                "max_depth": 5,                 # Limite la complexité des arbres
                "min_child_weight": 3,          # Réduit les splits inutiles
                "subsample": 0.8,               # Sous-échantillonnage des lignes
                "colsample_bytree": 0.95,        # Sous-échantillonnage des colonnes
                "n_estimators": 1000,            # Nombre d'arbres
                "reg_alpha": 0.005,              # Régularisation L1
                "reg_lambda": 1,               # Régularisation L2
                "gamma": 0.1,                    # Pénalisation des splits
                "random_state":42   # Fixe la seed pour la reproductibilité
            }

        # Initialisation du modèle XGBoost
        self.xgb_classifier = XGBClassifier()
    
    def train(self):
        # Division des données en jeu d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        X_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) 
        
        # Entraînement du modèle XGBoost
        self.xgb_classifier.fit(X_scaled, y_train)
        
        # Prédiction sur le jeu de test
        y_pred = self.xgb_classifier.predict(X_test_scaled)
        
        # Affichage de la précision
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        
        return accuracy_score(y_test, y_pred)
        
    def prediction(self, path):
        # Lancer l'entraînement avant les prédictions
        self.train()
        
        df_test = self.preprocess.get_preprocessed_test_input(path)

        # Standardisation des données de test
        X_test_scaled = self.scaler.transform(df_test.drop(columns=['ID']).values)

        # Prédiction avec le modèle entraîné
        y_pred = self.xgb_classifier.predict(X_test_scaled)
        
        # Sauvegarde des prédictions
        pred_df = pd.DataFrame()
        pred_df['ID'] = df_test['ID']
        pred_df['EventType'] = y_pred
        
        path_file = path + "/xgb_predictions.csv"
        pred_df.to_csv(path_file, index=False)
        
        print(f"Le fichier CSV a été sauvegardé avec succès sous : {path_file}")
    
    def best_param(self):
        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Mise à l'échelle des données
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Définition des hyperparamètres à explorer
        param_distributions = {
            'n_estimators': [100, 250, 500, 750, 1000],
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'learning_rate': [0.04, 0.05, 0.06],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.75, 0.9, 0.95],
            'reg_lambda': [1, 5, 10],
            'reg_alpha': [0.001, 0.005, 0.01],
            'min_child_weight': [1, 2, 3, 4, 5],
            'gamma': [0.1, 0.5, 1],
        }

        # Initialisation du classifieur XGBoost
        xgb = XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")

        # Initialisation de RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_distributions,
            scoring="roc_auc",  # Utilisation de l'AUC comme métrique
            cv=5,               # Validation croisée à 5 folds
            n_iter=50,          # Nombre d'itérations aléatoires
            verbose=1,
            random_state=42,
            n_jobs=-1           # Utilisation de tous les cœurs disponibles
        )

        # Entraînement de la recherche d'hyperparamètres
        random_search.fit(X_train_scaled, y_train)

        # Affichage des meilleurs résultats
        print("Meilleurs paramètres :", random_search.best_params_)
        print("Meilleur score AUC :", random_search.best_score_)

        # Sauvegarder les meilleurs hyperparamètres dans l'instance du modèle
        self.xgb_classifier = random_search.best_estimator_


    def analysis(self):
        from sklearn.feature_selection import SelectKBest, f_classif
        from xgboost import plot_importance
        import matplotlib.pyplot as plt

        # Séparer les caractéristiques (X) et les labels (y)
        df_X = self.df_train.drop(columns=["EventType", 'ID'])  # Toutes les colonnes sauf "EventType"
        df_y = self.df_train["EventType"]

        # Sélection des 10 meilleures caractéristiques
        selector = SelectKBest(score_func=f_classif, k=10)
        X_new = selector.fit_transform(df_X, df_y)

        # Récupérer les noms des colonnes sélectionnéesplot_importance(self.xgb_classifier, max_num_features=10)
        selected_features = df_X.columns[selector.get_support(indices=True)]

        print("Selected features:")
        print(selected_features)

        # Entraîner le modèle XGBoost avec les nouvelles caractéristiques
        self.xgb_classifier.fit(X_new, df_y)

        # Assigner les noms des caractéristiques sélectionnées au booster
        self.xgb_classifier.get_booster().feature_names = list(selected_features)

        # Visualiser l'importance des caractéristiques
        plot_importance(self.xgb_classifier, max_num_features=10)
        plt.show()
        
    def gridsearch(self):
        from sklearn.model_selection import GridSearchCV

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Mise à l'échelle des données
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        param_distributions = {
            'n_estimators': [100, 250, 500, 750, 1000],
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'learning_rate': [0.04, 0.05, 0.06],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.75, 0.9, 0.95],
            'reg_lambda': [1, 5, 10],
            'reg_alpha': [0.001, 0.005, 0.01],
            'min_child_weight': [1, 2, 3, 4, 5],
            'gamma': [0.1, 0.5, 1],
        }

        # Initialisation du classifieur XGBoost
        xgb = XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")

        grid_search = GridSearchCV(
                                estimator=xgb,
                                param_grid=param_distributions,
                                scoring="roc_auc",  # Utilisation de l'AUC comme métrique
                                cv=5,               # Validation croisée à 5 folds
                                verbose=1,
                                n_jobs=-1 )

        grid_search.fit(X_train_scaled, y_train)

        print("Meilleurs paramètres : ", grid_search.best_params_)
        print("Meilleur score AUC : ", grid_search.best_score_)
