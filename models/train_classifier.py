# import standard libraries
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import pickle
import time
import sys

# import ML libraries

import sklearn
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.datasets import make_multilabel_classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''Load data and split into X matric and y vector'''
    engine = create_engine('sqlite:///../data/'+database_filepath)
    df = pd.read_sql('SELECT * FROM master', engine)
    X = df["message"].values
    y = df.drop(["id", "message", "original", "genre"], axis=1).values
    category_names = df.drop(["id", "message", "original", "genre"], axis=1).columns
    return X, y, category_names

def tokenize(text):
    """a tokenization function to process our text data, which is splitting text into words / tokens"""
    tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')
    tokens = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def evaluate_model(model, X_test, y_test, category_names):
    '''Print out evaluation of trained model on test data'''
    y_pred = model.predict(X_test)
    # Todo:  iterating through the category columns
    index = 0
    for category in category_names:
        print("output category in column {}: {}".format(index, category))
        evaluation_report = classification_report(y_test[:,index], y_pred[:,index])
        index += 1
        print(evaluation_report)

def build_model():
    """building model pipeline for feature prediction using best params defined with the grid_search function"""
    database_filepath, model_filepath = sys.argv[1:]
    X, y, category_names = load_data(database_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)
    classifier = ExtraTreesClassifier(36)
    my_model = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize)),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(classifier, n_jobs=-1))
		])
    my_model.fit(X_train, y_train)
    evaluate_model(my_model, X_test, y_test, category_names)
    return my_model

"""If you would like to see the results of the Grid_Search evaluation, check the ML Pipeline Preparation jupyter notebook.
The function output gave the parameters which produced a model with a lower score, hence not used here."""

def save_model(my_model):
    """Function to export model as a pickle file"""
    filename = 'classifier.pkl'
    pickle.dump(my_model, open(filename, 'wb'))

def main():
    """main function"""
    my_model = build_model()
    save_model(my_model)
    print('Trained model saved!')

if __name__ == '__main__':
    main()