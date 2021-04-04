This README is for the Disaster Response Pipeline Project in Udacity.

There are thre main steps:
1. Loading and cleaning of the data and saving data in a db-file.
Run the following commands in the project's root directory to set up your database and model.
    For ETL pipeline that cleans data and stores in database:
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. Loading the data from the db file and training a model with an evaluation of that model
Run the following commands in the project's root directory to train your classifier, save it and the evaluate the model:
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3. Run this model in an application
Run the following command in the app's directory to run your web app:
    python run.py

Go to http://0.0.0.0:3001/ or https://view6914b2f4-3001.udacity-student-workspaces.com/
and type in the message you want to classify.
I tried: "Help me! Everywhere is fire."

Resoruces:
https://www.kaggle.com/depture/multiclass-and-multi-output-classification
https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
https://scikit-learn.org/stable/modules/multiclass.html
https://www.saltycrane.com/blog/2007/09/how-to-sort-python-dictionary-by-keys/



TODO:
- Descriptions
- Visuals on Web page


