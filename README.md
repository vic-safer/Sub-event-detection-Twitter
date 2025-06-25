We explain here how to use our code to create preprocessed datasets before giving it to a classifier.

Note : each notebook of the preprocessing pipeline gives a certain dataframe in .csv format, with a primary key 'ID'
It is then possible to create new dataframes by selecting the features you want from the several dataframes, merging by using the key
The ones we kept to classify is explained in the written report

Preprocessing_NLTK_vectors.ipynb :

This notebook transforms the series of 16 matches into a single dataframe of 2137 periods, and the 4 test matches in a single dataframe of 516 periods
Each period is represented by a vector with 206 dimensions in which you have 200 GloVe coordinates and 5 features (label, number of tweets of the period, number of tweets of the period + number of RT of tweets coming from the period, PeriodID, ID, growth rate of the number of tweets). 
Some of these features are just here to have a look at the data, they are not supposed to be used for a classification
path_xxxx variables : change the paths of the raw data and the final paths where you want to find your result
final name for the training dataset : df_train_NLTK.csv
final name for the training dataset : df_test_NLTK.csv


Preprocessing_features_USE.ipynb :

This notebook transforms the series of 16 training matches into a single dataframe of 2137 periods, and the 4 test matches in a single dataframe of 516 periods
path_xxxx variables : change the paths of the raw data
Each period is represented by a vector of 520 dimensions in which you have 512 USE coordinates and 8 features (label, normalized number of tweets of the period, PeriodID, ID, normalized number of emojis of the period, normalized number of excessive punctuation of the period, ratio of uppercase of the period, growth rate of the number of tweets)
final name for the training dataset : df_by_period_nuit.csv
final name for the test dataset : df_test_nuit.csv


Test.ipynb :
Note : you need the dataframe with the score_cosine, the dataframe given by Preprocessing_features_USE.ipynb

This notebook is here to test the preprocessed dataset with a classifier


Folder code_clean:
To use the various methods implemented in this folder, run the main.py file and call the desired function specified in the following line of code:
if __name__ == '__main__':
    example()

The methods in the main.py file invoke the various classes defined in the other files to preprocess and analyze the challenge data.
For building the training DataFrames, the main function calls CSV construction methods from individual files, each dedicated to a single feature: sentiment.py and keywords.py. Once this is done, the main function does not need to be called again.
preprocess.py
This file implements the Preprocess class, which retrieves the previously constructed CSV files (only once to save time and resources) and applies PCA to the data. For the training data, each training function calls the get_preprocessed_input method to obtain a DataFrame containing the model's input data. Similarly, the get_preprocessed_test_input method is used for test data.
sentiment.py
This file implements the Sentiment class, which is dedicated to measuring sentiment during specific periods.
To construct the DataFrame with sentiment metrics, call the build_csv_features function, which takes as input a preprocessed DataFrame containing the tweets. This function uses sentiment_features_batch, which processes a list of tweets and a threshold, and returns a list containing all the measured parameters.
The demo function is used to better understand the sentiment measurements on sample tweets, allowing fine-tuning of the parameters and functions.
keyword.py
This file implements a class for measuring the semantic distance between a tweet and a set of reference keywords associated with important time periods, generated using ChatGPT.
The get_keyword_distance function takes as input a DataFrame containing tweets organized by ID. For each ID, the Tweet column contains a list of preprocessed tweets. This function returns a DataFrame with the average proximity score over a period based on the keyword list.
The demo function is also included to better understand and refine the methods used.

svm.py, logreg.py, neuralnetwork.py
The SVMModel, LogisticRegressionClassifier, and NeuralNetwork classes all work in a similar manner, with train and prediction functions.
When initialized, these classes configure the model parameters and retrieve preprocessed data using the Preprocess class. The train function trains the model using the object's parameters and data. The prediction function automatically calls train to optimize the model and generates a CSV file with predictions on the test data.
xgboost_model.py
The XGBoostModel class works similarly to the other classes but has demonstrated more promising preliminary results.
Thus, additional functions have been implemented, such as:
best_param: Uses RandomizedSearchCV to find the best hyperparameters.
gridsearch: Similar to best_param but uses GridSearchCV from scikit-learn.
analysis: Visualizes the most important features for classification.

Note :
To invoke any function from the model classes, you need to create a function in main.py and instantiate an object of the class beforehand. For example:
def xgboost_pred(): 
model = XGBoostModel(62) 
acc = model.train() 
# model.prediction('/home/axfrl/Documents/X_3A/INF554/projet_twitter/Projet Twitter')
model.gridsearch() 
print("Accuracy: ", acc)



