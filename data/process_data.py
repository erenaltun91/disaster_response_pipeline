import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'



def load_data(messages_filepath, categories_filepath):
    """
    Loads data from .csv-files, extracts needed data and modifies it to get the needed columns.
    
    INPUT:
    messages_filepath - path of the message-file
    categories_filepath - path of the categories-file
    
    OUTPUT:
    df - dataframe with merged data from both .csv-files and one-hot encoded categories
    
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath,sep=',',engine='python')
    # load categories dataset
    categories = pd.read_csv(categories_filepath,sep=',',engine='python')
    
    # merge datasets
    messages.id.astype(int)
    categories.id.astype(int)
    df = messages.merge(categories,how='inner',on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
 
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.slice(stop=-2)

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    

    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    return df
    
    
    
    


def clean_data(df):
    """
    Cleans data from dataframe by dropping duplicates, nan-values and values which are not suitable for one-hot encoding
    
    INPUT:
    df - dataframe loaded before
    
    OUTPUT:
    df - cleaned dataframe
    
    """
    # check number of duplicates
    duplicated_count = df.duplicated().sum()
    
    if duplicated_count > 0:
        # drop duplicates
        df = df.drop_duplicates()
    else:
        print("No duplicates")
    
    # drop nan for prediction
    df = df.dropna(axis=0)
    
    # drop all rows with a number other than 0 or 1
    df = df.drop(df[df['related'] == 2].index,axis=0)
    
    
    
    return df


def save_data(df, database_filename):
    """
    Saves dataframe to a .db-file
    
    INPUT:
    df - dataframe loaded and cleaned before
    database_filename - path to save .db-file with dataframe
    
    OUTPUT:
    database as .db-file 
    
    """
    # create engine for database connection
    engine = create_engine('sqlite:///'+database_filename)
    
    # save dataframe to database file
    df.to_sql('Table', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()