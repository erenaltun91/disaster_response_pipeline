This README is for the Disaster Response Pipeline Project in Udacity.

There are thre main steps:
1. Loading and cleaning of the data and saving data in a db-file.
Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. Loading the data from the db file and training a model with an evaluation of that model
    - To run ML pipeline that trains classifier and saves
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3. Run this model in an application

