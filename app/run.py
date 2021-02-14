import json
import pandas as pd
import pickle
import joblib as joblib
import plotly
import plotly.express as px
import collections
import nltk
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from pandas import DataFrame


app = Flask(__name__)

clean_tokens = []
def tokenize(text):
    """a tokenization function to process our text data, which is splitting text into words / tokens"""
    tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')
    tokens = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('master', engine)
stop_words = stopwords.words('english')
clean_tokens = tokenize(df['message'].to_string())

# load model
model = joblib.load("classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    pivoted = pd.melt(df,id_vars = ['id'],value_vars=df.drop(['id','genre','message','original'], axis=1), var_name='category' )
    pivoted_grouped = pivoted.groupby('category',as_index=False).sum()
    top15 = pivoted_grouped.sort_values(by='value', ascending=False)[:15]
    bottom15 = pivoted_grouped.sort_values(by='value', ascending=True)[:15]

    word_counts = Counter(x for x in clean_tokens if x not in stop_words)
    most_common_words = word_counts.most_common(15)
    df_words = DataFrame(most_common_words, columns=['words', 'counts'])

    # create visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x=genre_names,
                    y=genre_counts,
                    opacity=0.5
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'size': 16,
                'yaxis': {
                    'title': "Count of Messages"
                },
                'xaxis': {
                    'title': "Genre",
                        'size': 16,
                        'color': '#003366'
                }
            }
        },

        {
            'data': [
                Bar(
                    x=top15['category'],
                    y=top15['value']
                )
            ],

            'layout': {
                'title': '15 Categories with HIGHEST Count of Messages',
                'yaxis': {
                    'title': "Count of Messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=bottom15['category'],
                    y=bottom15['value']
                )
            ],

            'layout': {
                'title': '15 Categories with LOWEST Count of Messages',
                'yaxis': {
                    'title': "Count of Messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_words['words'],
                    y=df_words['counts'],
                opacity = 0.5
                )
            ],

            'layout': {
                'title': 'Most Common Words Found in Messages',
                'yaxis': {
                    'title': "Frequency of Words in Messages"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()