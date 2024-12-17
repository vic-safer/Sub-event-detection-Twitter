from sentence_transformers import SentenceTransformer, util # type: ignore
from collections import Counter
import nltk # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
import string
import pandas as pd # type: ignore
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore

nltk.download('punkt')  # Télécharger les ressources pour la tokenisation
nltk.download('punkt_tab')

class KeywordSemanticMatcher:
    def __init__(self, model_name="all-MiniLM-L6-v2", similarity_threshold=0.4):
        """
        Initialise la classe avec une liste de mots-clés et un modèle de sentence-transformers.

        :param keywords: Liste de mots-clés à surveiller.
        :param model_name: Nom du modèle pré-entraîné Sentence-Transformers.
        :param similarity_threshold: Seuil de similarité pour reconnaître les mots proches.
        """
        self.keywords = [
            # Réactions positives
            "goal", "golazo", "amazing", "awesome", "fantastic", "incredible", "legendary", 
            "brilliant", "great", "superb", "banger", "unstoppable", "masterpiece", 
            "beautiful", "perfect", "top corner", "hat-trick", "what a goal", 
            "dream goal", "freekick", "great shot", "magic", "unstoppable shot", 
            "clutch moment", "game-changer", "stunner", "epic", "winner", "decisive", 
            "comeback", "equalizer", "what a save", "clean sheet", "man of the match", 
            "golden goal", "super sub", "dominant performance", "glory", "champion", 
            "vamos", "olé", "olééé", "celebration", "win"

            # Réactions négatives
            "offside", "penalty missed", "red card", "yellow card", "injury", "what a miss", 
            "huge mistake", "blunder", "howler", "disaster", "shocking", "awful", 
            "terrible", "poor", "missed opportunity", "own goal", "disgraceful", 
            "controversial decision", "VAR", "boo", "bad tackle", "foul", "flop", 
            "bottled", "disappointment", "failure", "poor defense", "bad pass", 
            "unlucky", "heartbreaking", "choke", "off the post", "waste", "error", 
            "penalty conceded", "outplayed", "undeserved", "unfair", "robbed", "dirty play", 
            "time-wasting",

            # Réactions émotionnelles
            "wow", "woah", "omg", "unbelievable", "insane", "crazy", "madness", 
            "jaw-dropping", "electric", "tense", "nail-biting", "ecstatic", "heartbreaking", 
            "relief", "despair", "tears", "joy", "crying", "cheering", "clapping", 
            "shouting", "screaming", "booing", "disbelief", "euphoria", "passion", 
            "nervous", "stress", "shock", "surprise", "anger", "rage", "frustration",

            # Événements spécifiques au football
            "kickoff", "halftime", "fulltime", "penalty", "freekick", "corner", "header", 
            "assist", "cross", "volley", "deflection", "counterattack", "clearance", 
            "block", "interception", "pressing", "possession", "build-up", "through ball", 
            "last-minute goal", "added time", "stoppage time", "extra time", "penalty shootout", 
            "set piece", "tackle", "handball", "VAR check", "final whistle", "celebration", 
            "pitch invasion", "fans chanting", "crowd roar", "stadium explosion", 
            "underdog victory", "giant killing", "rivalry clash", "derby",

            # Mots d'encouragement
            "yes", "let's go", "c'mon", "keep it up", "fight", "believe", "stay focused", 
            "push", "never give up", "all in", "on fire", "unstoppable", "unstoppable force", 
            "keep going", "stronger together", "one team", "united", "unstoppable team", 
            "amazing spirit", "teamwork",

            # Réactions culturelles et internationales
            "vamos", "olé", "forza", "allez", "hup", "auf geht's", "viva", "davai", 
            "arriba", "vamossss", "siuuuu", "si", "siiiiii", "calma", "tranquilo", 
            "gagné", "won", "victoria", "gloria", "campeones"
        ]

        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.keyword_embeddings = self.model.encode(self.keywords, convert_to_tensor=True)

    def find_similar(self, text, norm):
        """
        Trouve les mots-clés sémantiquement proches d'un texte donné.

        :param text: Texte à analyser.
        :return: Liste des mots-clés sémantiquement proches.
        """

        self.model.similarity_fn_name = norm

        # Encoder le texte
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        
        # Calculer la similarité cosinus entre le texte et les mots-clés
        # Compute cosine similarities
        similarity_matrix = self.model.similarity(text_embedding, self.keyword_embeddings)
        
        max_scores, max_indices = similarity_matrix.max(dim=1)
        
        return max_scores

    def get_keyword_distance(self, df):
        
        norm_list = ['cosine']
        max_scores = {"cosine": []}

        for tweets in tqdm(df['Tweet'].tolist()):
            # Encode les tweets en embeddings
            tweet_embeddings = self.model.encode(tweets, convert_to_tensor=True)
            
            for norm in norm_list:
                
                # Calcul de la similarité cosinus
                similarity_matrix = self.model.similarity(tweet_embeddings, self.keyword_embeddings)

                # Trouve le score maximum pour chaque tweet
                max_score, _ = similarity_matrix.max(dim=1)  # max_scores : Tensor 1D
                
                # Applique le seuil et calcule la moyenne des scores retenus
                scores = max_score.cpu().numpy()  # Convertit en NumPy
                
                high_scores = np.where(scores > self.similarity_threshold, 1, 0)
                
                sum_score = np.sum(high_scores)

                max_scores[norm].append(sum_score)
                
        result_df = pd.DataFrame({
            'ID': df['ID'],
            'score_cosine': max_scores["cosine"]
            #'score_dot': max_scores["dot"],
            #'score_euclidean': max_scores["euclidean"],
            #'score_manhattan': max_scores["manhattan"]
        })
        
        print("end of the keyword distance processing")

        return result_df
        
    def demo(self):
        tweets = [
            '#ARGBEL 🙌🙌🙌argentina for the win🙌🙌🙌',
            'Let s go Argentina !!!',
            'It was coming. GOALLLLLLLLLLLLLLLLLLLLLLLLLLL.',
            'And Argentina strikes first',
            'Argentina is to good',
            'I m following Argentina versus Belgium in the FIFA Global Stadium #ARGBEL #worldcup #joinin http://t.co/v0Uh6Bqpdo perfect',
            'Belgium always has the other team s flag on their jerseys. aw'
        ]
        
        tweet_embeddings = self.model.encode(tweets, convert_to_tensor=True)
        
        similarities = self.model.similarity(tweet_embeddings, self.keyword_embeddings)

        # Trouve le score maximum pour chaque tweet
        max_score, max_index = similarities.max(dim=1)  # max_scores : Tensor 1D
        # Output the pairs with their score
        for idx_i, sentence1 in enumerate(tweets):
            print(sentence1)
            
            print(f" - {self.keywords[max_index[idx_i]]: <30}: {similarities[idx_i][max_index[idx_i]]:.4f}")
                        