




TODO:
- Descriptions
- Visuals on Web page

# Project 3 of the Data Scientist Nanodegree Program - Disaster Response Pipeline
This project loads data about disaster messages, displays visual on a web page and estimates the category of the message, which will also displayed on a web page.

## Installations
- Put the files in this repo into a folder of your choice
- Install following python libraries:
    - pandas, numpy, matplotlib
    
`pip install numpy`  
`pip install pandas`  
`pip install matplotlib`

## Execution
- Set your chosen folder as the workingspace
- 1. Loading and cleaning of the data and saving data in a db-file.
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

- 2. Loading the data from the db file and training a model with an evaluation of that model
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- 3. Run this model in an application
    `python app/run.py`

- Go to http://0.0.0.0:3001/ or https://view6914b2f4-3001.udacity-student-workspaces.com/ and type in the message you want to classify.
    - I tried: "Help me! Everywhere is fire."


## Project Motivation
- In the course of the Data Scientist Nanodegree Prgram this analysis was done to be published on a social media platform ('Medium')
- It compares the participation on the survey of countries and genders between the years
- It compares the working hours per week and the age demographics in 2020 between the countries  

## File Descriptions

- Project 1 - Survey Analysis.ipynb
    - Main file with code
    - Starts analysis and saves plot into folder
    
- functions.py
    - Module with functions for the analysis
    
- X_survey_results.csv
    - Survey Data from https://insights.stackoverflow.com/survey
    - X: years from 2017 to 2020
    
- survey_results_schema.csv
    - Description of data
    
## Usage
- Run the Code in the ipynp-file
- The data will be read in from the 'data' directory
- The plots will be saved in the 'results' directory 

## Summary
- The analysis shows that the USA has the highest amount of participants and that the trend about the participation on the survey does not change
- The data on the genders shows, that the participants are mostly men. The other genders besides females decreased over the years
- For working hours per week you should choose Germany because it has the least number of working hours with an average of 39 working hours per week.
- The age demographics plots show that India has the youngest participants on average.

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
