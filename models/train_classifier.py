# import standard libraries
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import pickle
import time

# import ML libraries

import sklearn
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.datasets import make_multilabel_classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql('SELECT * FROM master', engine)
df.head(3)

# define and format X and y
X = df['message'].to_string()
y = df.drop(['id', 'message','original','genre'], axis=1)


def load_data():
    '''Load data and split into X matric and y vector'''

    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('SELECT * FROM master', engine)
    X = df["message"].values
    Y = df.drop(["id", "message", "original", "genre"], axis=1).values
    return X, y

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

def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)

# find classifier that fits best

"""
def find_classifier():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    classifiers = [
        KNeighborsClassifier(36),
        DecisionTreeClassifier(),
        RandomForestClassifier(36),
        ExtraTreeClassifier(),
        ExtraTreesClassifier(36),
        RadiusNeighborsClassifier(36)
    ]

    for classifier in classifiers:
        pipe = Pipeline(steps=[
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier)])

        pipe.fit(X_train, y_train)
        print(classifier)
        print("model score: %.3f" % pipe.score(X_test, y_test))


def build_model():
    #building model pipeline for feature prediction using the best score classifier aka clf,
       # based on the output of the find_classifier function
       # Classification report terminology:
       # The recall means "how many of this class you find over the whole number of element of this class"
      #  The precision will be "how many are correctly classified among that class"
      #  The f1-score is the harmonic mean between precision & recall
      #  The support is the number of occurence of the given class in your dataset
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    classifier = ExtraTreesClassifier(36)  # use another classifier with best score accordingly
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(classifier, n_jobs=-1))
    ])

    # train classifier
    pipeline.fit(X_train, y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test)

    # display_results(y_test, y_pred)
    # print(f"x_train:{X_train.shape}"),print(f"x_test: {X_test.shape}") , print(f"y_train: {y_train.shape}"), print(f"y_test:{y_test.shape}")
    print(classification_report(y_test, y_pred))

#Use grid search to find better parameters.

def grid_search():
    #Using grid search to find better parameters
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    classifier = ExtraTreesClassifier(36)  # use another classifier with best score accordingly

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(classifier, n_jobs=-1))
    ])

    parameters = {'clf__estimator__criterion': ["gini", "entropy"],
                  'clf__estimator__n_jobs': [-1, 1],
                  'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
                  'clf__estimator__max_depth': [2, 4, 5, 6, 7, 8]}

    cv = GridSearchCV(
        pipeline,
        parameters,
        n_jobs=1
    )

    cv.fit(X_train, y_train)

    # return cv
    print(cv.best_params_)
    print(cv.best_score_)
"""

def build_model2():
    """building model pipeline for feature prediction using best params defined with the grid_search function"""

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    classifier = ExtraTreesClassifier(36, criterion='gini', max_depth=2, max_features='auto', n_jobs=-1)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(classifier))
    ])

    # train classifier
    pipeline.fit(X_train, y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test)

    # display results
    # display_results(y_test, y_pred)
    # print(f"x_train:{X_train.shape}"),print(f"x_test: {X_test.shape}") , print(f"y_train: {y_train.shape}"), print(f"y_test:{y_test.shape}")
    print(classification_report(y_test, y_pred))
    display_results(y_test, y_pred)

"""Exporting the final model into a pickle file"""

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)
classifier = ExtraTreesClassifier(36, criterion='gini', max_depth=2, max_features='auto', n_jobs=-1)

model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(classifier, n_jobs=-1))
    ])
model.fit(X_train,y_train)

filename = 'classifier.pkl'
pickle.dump(model, open(filename, 'wb'))