import pandas as pd # type: ignore
import os
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from tqdm import tqdm # type: ignore
import re
from sklearn.dummy import DummyClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from methods import Preprocess, Sentiment, KeywordSemanticMatcher, LogisticRegressionClassifier, NeuralNetwork, XGBoostModel, SVMModel, AEModel, BertFineTuning, BertModel

folder_path = "/home/axfrl/Documents/X_3A/INF554/projet_twitter/code_clean"

def main():
    train_path = folder_path + "/data/challenge_data/eval_tweets"
    dataframes = []

    for fichier in os.listdir(train_path):
        if fichier.endswith('.csv'):
            chemin_complet = os.path.join(train_path, fichier)
            df = pd.read_csv(chemin_complet)
            dataframes.append(df)

    df_train = pd.concat(dataframes, ignore_index=True)
    
    preprocessor = Preprocess()

    df_train.dropna(inplace=True)
    df_train = preprocessor.url_removal(df_train)
    df_train = df_train[~df_train['Tweet'].str.startswith('RT')]
    grouped_df = df_train.groupby('ID').agg({
        'Tweet': lambda tweets: ' // '.join(tweets)  # Concatène les tweets avec " // " comme séparateur
    }).reset_index()
    print(grouped_df)
    df_11 = df_train[df_train["MatchID"]==11] #df du premier match
    print("le batch de tweet est de taille ", len(df_train))
    
    grouped_df = df_train.groupby('ID').agg({
        'Tweet': list,          # Crée une liste des tweets pour chaque ID
    }).reset_index()

    #print(df)
    sentiment_model = Sentiment()
    keyword_model = KeywordSemanticMatcher()
    
    save_csv_features(grouped_df)
    
    return True

def build_emo_file(file_path= folder_path+"/data/challenge_data/features_emo_trainset2.csv"):
    sent_model = Sentiment()
    train_path = folder_path+"/data/challenge_data/train_tweets"
    dataframes = []

    for fichier in os.listdir(train_path):
        if fichier.endswith('.csv'):
            chemin_complet = os.path.join(train_path, fichier)
            df = pd.read_csv(chemin_complet)
            dataframes.append(df)

    df_train = pd.concat(dataframes, ignore_index=True)
    
    preprocessor = Preprocess()

    df_train.dropna(inplace=True)
    df_train = preprocessor.url_removal(df_train)
    df_train = df_train[~df_train['Tweet'].str.startswith('RT')]
    grouped_df = df_train.groupby('ID').agg({
        'Tweet': list,          # Crée une liste des tweets pour chaque ID
    }).reset_index()
    
    emo_df = sent_model.build_csv_features(grouped_df)
    emo_df.to_csv(file_path, index=False)  # index=False pour ne pas inclure l'index dans le CSV
    
    print(f"Le fichier CSV a été sauvegardé avec succès sous : {file_path}")
    
def correlation_matrix(df_data, df_label):
    
    df_combined = pd.merge(df_data, df_label, on='ID', how='inner')
    
    correlation_matrix = df_combined.corr()
    
    print(correlation_matrix)
    
    return df_combined

def save_csv_features(data, file_path=folder_path+"/data/challenge_data/features_testset.csv"):
    sentiment_model = Sentiment()
    df_sentiment =sentiment_model.build_csv_features(data)
    
    keyword_model = KeywordSemanticMatcher()
    df_keyword  =keyword_model.get_keyword_distance(data)
    
    df_combined = pd.merge(df_sentiment, df_keyword, on='ID', how='inner')
    
    # Sauvegarder en fichier CSV
    df_combined.to_csv(file_path, index=False)  # index=False pour ne pas inclure l'index dans le CSV
    
    print(f"Le fichier CSV a été sauvegardé avec succès sous : {file_path}")

def log_reg():
    model = LogisticRegressionClassifier()
    model.train()
    #model.prediction(folder_path+'/data')

def xgboost_pred():
    model = XGBoostModel(62)
    acc = model.train()
    #model.prediction(folder_path+'/data')
    model.gridsearch()
    print("accuracy: ", acc)
    
def NN():
    neural_net = NeuralNetwork()
    #neural_net.train()
    neural_net.prediction(folder_path+'/data')

def svm():
    model = SVMModel()
    model.prediction(folder_path+'/data')

if __name__ == '__main__':
    main()
