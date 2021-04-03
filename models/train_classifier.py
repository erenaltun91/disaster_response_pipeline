import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])
import os
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """
    Loads data from .db-file and extracts needed data.
    
    INPUT:
    database_filepath - path of the .db-file
    
    OUTPUT:
    X - the messages column, which should be used to train the calssifier  
    Y - the categories/classes columns, which should be used to train the classifier 
    category_names - name of the categories/classes
    
    
    """
    # create engine for database connection
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    cur = engine.connect()
    # read table as dataframe
    df = pd.read_sql_table('Table', engine)
    # messages as values for the X
    X = df.message.values
    # categories as values for the Y
    Y = df.loc[:,'related':].values
    
    
    category_names = list(df.loc[:,'related':].columns)
    
    return X,Y,category_names


def tokenize(text): 
    """
    Converts text to clean tokens.
    
    INPUT:
    text - text or message 
    
    OUTPUT:
    clean_tokens - list of modified words
    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Builds model pipeline.
    
    INPUT:
    
    
    OUTPUT:
    model_pipeline
    
    """
    #initialize pipeline
    pipeline =  Pipeline([

        ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
        ('clf' , MultiOutputClassifier(RandomForestClassifier()))    
    ])
    
    #set parameters for GridSearchCV-method
    parameters = {
    'clf__estimator__criterion':['gini','entropy'],
    'clf__estimator__max_features':['auto','sqrt','log2'] 
    }

    model_pipeline = GridSearchCV(pipeline,param_grid=parameters)
    
    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate your model with predicting the classes and test against the true classes with specific measures.
    INPUT:
    model - trained model
    X_test - message values of the testset
    Y_test - true classes used to compare predicted values
    category_names - names of the categories/classes to be estimated
    
    OUTPUT:
    Prints out classification report for each column
    
    """
    #predict values
    predicted = model.predict(X_test)
    #print out classification report for each column
    c=0
    for column in category_names:
        print(column)
        print(classification_report(Y_test[c],predicted[c]))
        c = c+1

    


def save_model(model, model_filepath):
    """
    Save the model to a .pickle-file
    
    INPUT:
    model - trained and evaluated model
    model_filepath - path to save the .pickle-file
    
    OUTPUT:
    .pickle-file
    
    """
    # save model as pickle-file
    pickle.dump(model, open(model_filepath, 'wb'))


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