## Building_Pipelines
Project involving building an **ETL Pipeline, ML Pipeline** as well as a simple **Flask Web App**.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run *ETL pipeline* that cleans data and stores in database
        `python process_data.py "messages.csv" "categories.csv" "DisasterResponse.db"'
		It loads the datasets, fixes the data by resolving quality and tidiness issues, and loads the result into an SQLite database.
		
    - To run *ML pipeline* that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`