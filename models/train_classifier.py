import sys
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

import sqlite3
import pandas as pd
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.utils import parallel_backend
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBClassifier


stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    """ Loads pre-processed data from supplied file path
        Returns X and Y datasets and a list of category names
    """
    
    cxn = sqlite3.connect(database_filepath)
    df = pd.read_sql('select * from messages_cleaned', cxn)
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    category_names = list(y.columns)

    return X , y, category_names

def tokenize(text):
    """
    Tokenizes, Lemmatizes and filters stop words
    Returns list of tokens 
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Instantiates model and returns gridsearch model of pipeline

    """

    with parallel_backend('multiprocessing'):
        print('Building model')
        pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(XGBClassifier(objective = 'binary:hinge')))
            ])

        parameters = {
            #'vect__ngram_range': ((1, 1), (1, 2)),
            #'vect__max_df': (0.5, 0.75),
            #'vect__max_features': (None, 5000, 10000),
            #'tfidf__use_idf': (True, False),
            #'clf__estimator__n_estimators': [10, 50],
            #'clf__estimator__max_depth': [10, 20]
            }

        return GridSearchCV(pipeline, param_grid=parameters,  cv=5, n_jobs=-1, scoring='f1_weighted')
        

def evaluate_model(model, X_test, Y_test, category_names):
    
    #Make predictions
    y_pred = pd.DataFrame(model.predict(X_test))
    
    scores = []
    for i in range(0,36):
        tmp=precision_recall_fscore_support(Y_test.iloc[:,i], y_pred.iloc[:,i], average='binary')
        scores.append(tmp)
    scores = pd.DataFrame(scores)
    scores.columns = ['precision','recall','fscore','support']
    scores.drop(columns = 'support',inplace = True)
    scores['categories'] = category_names
    scores.set_index('categories', inplace = True)
    scores = scores[scores.precision>0] #eliminating empty rows

    print('precision:%.2f, recall:%.2f, fscore:%.2f' % (scores.precision.mean(), scores.recall.mean(), scores.fscore.mean()))
    
    return


def save_model(model, model_filepath):
    """
    Saves model as pickel file to specified location
    """
    pkl = open(model_filepath, 'wb')
    pickle.dump(model, pkl)
    pkl.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()