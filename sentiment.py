from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import torch

# methode nltk
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from nltk import tokenize

import pandas as pd
from tqdm import tqdm

nltk.download('vader_lexicon')

class Sentiment:
    def __init__(self):
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.config = AutoConfig.from_pretrained(MODEL)
        # PT
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        
    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def sentiment_features_batch(self, sentences, threshold):
        # Prétraitement en batch
        sentences = [self.preprocess(sentence) for sentence in sentences]
        encoded_inputs = self.tokenizer(sentences, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        truncation=True,
                                        max_length=512  # Longueur maximale pour BERT
                                        )

        # Obtenir les prédictions du modèle en batch
        with torch.no_grad():  # Pas besoin de calculer les gradients
            outputs = self.model(**encoded_inputs)

        # Convertir les logits en scores de probabilité
        logits = outputs.logits.detach().numpy()
        scores = softmax(logits, axis=1)  # Calculer le softmax pour chaque batch

        # Extraire les scores individuels
        neg_scores = scores[:, 0]  # Scores négatifs
        neu_scores = scores[:, 1]  # Scores neutres
        pos_scores = scores[:, 2]  # Scores positifs

        # Calcul de l'intensité et du niveau d'émotion
        emotion_intensity = np.sqrt(neg_scores**2 + pos_scores**2)
        emotion_level = emotion_intensity / (emotion_intensity + neu_scores)

        # Filtrer les niveaux d'émotion au-dessus du seuil
        emotion_list = emotion_level[emotion_level >= threshold]
        
        # Moyennes des différentes dimensions
        mean_pos = np.mean(pos_scores)
        mean_neg = np.mean(neg_scores)
        mean_neu = np.mean(neu_scores)

        # Moyenne des niveaux d'émotion retenus
        mean_emotion = np.mean(emotion_list) if len(emotion_list) > 0 else 0

        # Générer les features finales
        features = np.array([mean_emotion, mean_pos, mean_neg, mean_neu])

        return features

    def build_csv_features(self, df_data):
        df = {'ID': [], 'mean': [], 'mean_pos': [], 'mean_neg': [], 'mean_neu': []}
        data = [(ID, sentences) for ID, sentences in zip(df_data['ID'].tolist(), df_data['Tweet'].tolist())]

        # Parcourir les données
        print("starting the emotions measurement")
        for ID, sentences in tqdm(data):
            feat = self.sentiment_features_batch(sentences, 0)
            df['ID'].append(ID)
            df['mean'].append(feat[0])
            df['mean_pos'].append(feat[1])
            df['mean_neg'].append(feat[2])
            df['mean_neu'].append(feat[3])

        # Construire la DataFrame à partir du dictionnaire
        df = pd.DataFrame(df)

        df['delta_mean'] = df['mean'].pct_change() * 100
        df['delta_mean_pos'] = df['mean_pos'].pct_change() * 100
        df['delta_mean_neg'] = df['mean_neg'].pct_change() * 100
        df['delta_mean_neu'] = df['mean_neu'].pct_change() * 100

        return df
        
    def demo(self, tweets):
        if tweets is None:
            liste_demo=[
                "GGOOOOAAAAALLLLL!!!!",
                "go Argentina",
                "Really hope #BRA put on a spectacular tonight👏👏",
                "Fred is shit! #cmr #bra",
                "Brazil already score! #BRA",
                "explained to grandma how to watch #bra-#cmr on her laptop :D fortunately goatd was still open after i watched #ned #chi"
            ]
        else:
            liste_demo = tweets
        
        def preprocess(text):
            new_text = []
            for t in text.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t
                new_text.append(t)
            return " ".join(new_text)
        
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        config = AutoConfig.from_pretrained(MODEL)
        # PT
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        #model.save_pretrained(MODEL)
        for text in tqdm(liste_demo):
            text = preprocess(text)
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            #print(text)
            #print(type(scores))
            #for i in range(scores.shape[0]):
            #    l = config.id2label[ranking[i]]
            #    s = scores[ranking[i]]
            #    print(f"{i+1}) {l} {np.round(float(s), 4)}")