# Project 3 of the Data Scientist Nanodegree Program - Disaster Response Pipeline
This project loads data about disaster messages, displays visual on a web page and estimates the category of the message, which will also displayed on a web page.

## Installations
- Put the files in this repo into a folder of your choice
- Install following python libraries:
    - pandas, numpy, matplotlib
    
`pip install numpy`  
`pip install pandas`  
`pip install matplotlib`

## Project Motivation
- In the course of the Data Scientist Nanodegree Prgram this analysis was done to classify messages from disasters
- Messages, which were already classified are used to train models to classify new messages
- Models have been tuned for better performance


## File Descriptions
- data/disaster_categories.csv
    - Categories of different disasters

- data/disaster_messages.csv
    - Messages of different disasters

- data/process_data.py
    - Loads and cleans data
    - Merges both csv-files
    - Saves data as database file
    
- models/train_classifier.py
    - Loads data from database file
    - Trains models and prints out evaluation metrics
    
- app/run.py
    - Building the web page
    - Using data to display visuals
    - Using model to predict categories of new messages

- master.html
    - Web page with visuals

- go.html
    - Web page with classification method

   
## Usage
- Opem the command terminal
- Set your chosen folder as the workingspace
- Run the following commands in terminal:
    - 1. Loading and cleaning of the data and saving data in a db-file  
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - 2. Loading the data from the db file and training a model with an evaluation of that model  
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    - 3. Run this model in an application  
    `python app/run.py`

- Go to http://0.0.0.0:3001/ or https://view6914b2f4-3001.udacity-student-workspaces.com/ and type in the message you want to classify  
    - I tried: "Help me! Everywhere is fire."

## Summary
- The most common category is 'related' and the least common category is 'child_alone'
- The evaluation of the model seem good
- The classification could be better

## Licensing, Authors, Acknowledgement

- This analysis was made by me, an Udacity student with the data provided by Udacity
- The results and the code can be used in any way
- Please contact me for feedback

### Contact: 
LinkedIn: https://www.linkedin.com/in/eren-altun91/


### Resources:  
https://www.kaggle.com/depture/multiclass-and-multi-output-classification  
https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680  
https://scikit-learn.org/stable/modules/multiclass.html  
https://www.saltycrane.com/blog/2007/09/how-to-sort-python-dictionary-by-keys/  
