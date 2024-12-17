import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from .sentiment import Sentiment
import os

folder_path = "/home/axfrl/Documents/X_3A/INF554/projet_twitter/code_clean"

class Preprocess:
    def __init__(self, dim_pca=None):
        path = folder_path+'/data/df_train_nuit_finale.csv'

        df_night = pd.read_csv(path)
        #cols_pca = [str(k) for k in range(200)]  # Colonnes de 0 à 199 (en noms de colonnes)
        X_pca_input = df_night.drop(columns=['ID', 'EventType'], axis=1).values  # Matrice d'entrée pour la PCA

        if dim_pca is None:
            self.n_pca_components=10
        else:
            self.n_pca_components=dim_pca
        # Appliquer une PCA pour réduire à 50 dimensions
        self.pca = PCA(self.n_pca_components)
        self.pca.fit(X_pca_input)
        
    def url_removal(self, df):
        """
        lit tous les tweet d'un data frame et enlève les tweets contenant les url
        """
        
        # Regex pour détecter les URLs
        url_regex = r"http[s]?://\S+"

        # Filtrer les tweets qui NE contiennent PAS d'URLs
        df_no_urls = df[~df["Tweet"].str.contains(url_regex, regex=True, na=False)]

        return df_no_urls
                    
    def get_preprocessed_input(self):
        path = folder_path+'/data/df_train_nuit_finale.csv'

        cols_pca = [str(k) for k in range(200)]  # Colonnes de 0 à 199 (en noms de colonnes)
        
        df_night = pd.read_csv(path)
        #cols_pca = [str(k) for k in range(200)]  # Colonnes de 0 à 199 (en noms de colonnes)
        X_pca_input = df_night.drop(['ID', 'EventType'], axis=1).values

        X_pca_reduced = self.pca.transform(X_pca_input)

        # Créer un DataFrame pour les résultats de la PCA avec des noms de colonnes explicites
        pca_columns = [f"pca_{i}" for i in range(self.n_pca_components)]  # Noms des nouvelles colonnes
        df_pca = pd.DataFrame(X_pca_reduced, columns=pca_columns, index=df_night.index)

        
        #df_non_pca = df_non_pca.drop(columns=['mean_pos', 'mean_neg', 'mean_neu', 'delta_mean', 'delta_mean_pos', 'delta_mean_neg', 'delta_mean_neu'])
        df_non_pca = df_night[['EventType', 'ID']]
        # Fusionner les résultats de la PCA avec les colonnes restantes
        df_pca_sana = pd.concat([df_non_pca, df_pca], axis=1)

        df_train = df_pca_sana.fillna(0)
        
        return df_train
    
    def get_preprocessed_test_input(self, path):
        testset_sana_path = path + '/data/df_test.csv'
        testset_emo_kw_path = path + '/data/keyword_emotion_features_testset.csv'
        testset_loss_path = path + '/data/df_test_loss.csv'
        
        df_loss_test = pd.read_csv(testset_loss_path)
        df_sana = pd.read_csv(testset_sana_path)
        df_emo_kw = pd.read_csv(testset_emo_kw_path)
        
        cols_pca = [str(k) for k in range(200)]  # Colonnes de 0 à 199 (en noms de colonnes)
        X_pca_input = df_sana[cols_pca].values  # Matrice d'entrée pour la PCA

        X_pca_reduced = self.pca.transform(X_pca_input)

        # Créer un DataFrame pour les résultats de la PCA avec des noms de colonnes explicites
        pca_columns = [f"pca_{i}" for i in range(self.n_pca_components)]  # Noms des nouvelles colonnes
        df_pca = pd.DataFrame(X_pca_reduced, columns=pca_columns, index=df_sana.index)

        # Conserver les autres colonnes de df_sana qui ne font pas partie de la PCA
        df_non_pca = df_sana.drop(columns=cols_pca)
        #df_non_pca = df_non_pca.drop(columns=['labels_pred'])
        #df_non_pca = df_non_pca.drop(columns=['mean_pos', 'mean_neg', 'mean_neu', 'delta_mean', 'delta_mean_pos', 'delta_mean_neg', 'delta_mean_neu'])

        # Fusionner les résultats de la PCA avec les colonnes restantes
        df_pca_sana = pd.concat([df_non_pca, df_pca], axis=1)
        
        df_test = pd.merge(df_pca_sana, df_emo_kw, on='ID', how='outer')
        
        df_test = df_test.fillna(0)
        
        return df_test