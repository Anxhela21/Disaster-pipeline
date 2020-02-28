# Disaster Response Pipeline Project

This project was focused around building a data dashboard that users can input a keyword regarding their state of emergency which would trigger a corresponding category. There are a few visualizations that depict the counts of each disaster category and they're originating source built using Plotly. 

This app is using Flask framework and run on Heroku.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### Acknowledgements #
Credit to Udacity for the starter codes and FigureEight for provding the data used by this project.
