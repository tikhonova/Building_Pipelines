### Project Description
</br> 
*Goals*: 
1. Analyze real messages that were sent during disaster events.
2. Create the best fitting machine learning model and build a pipeline for an API that classifies disaster messages. 
2. Build a web interface that would allow User to input a new message and get classification results in several categories. The web app will also display visualizations of the training data.
</br> 
*Dataset*: imported tables with messages and given categories.
</br> 
*Scripts* : 
* process_data.py that cleans data and stores in database
* train_classifier.py that trains and saves selected classifier
* run.py that runs a web app
</br> 
*Additional files*:
* SQLite database resulting from the first script
* PKL file resulting from the second script
* Jupyter Notebooks with exploratory analys and other work done prior to building scripts
* Templates that help run the web app.
</br> 
### Instructions:
1. Set up your database and model: run the following commands in the project's root directory to load libraries and set up your database and model.

    1.1. Run the *process_data.py* script that cleans data and stores in database: `python process_data.py "messages.csv" "categories.csv" "DisasterResponse.db"'
	The above command loads the datasets, fixes the data by resolving quality and tidiness issues, and loads the result into an SQLite database.
		
    1.2. Run the *train_classifier.py* script that trains classifier and saves it as a pickle file: `python train_classifier.py DisasterResponse.db classifier.pkl`
	The second command builds and saves the model into a .pkl file. 
	
	1.3. For additional information, see the *ETL pipeline* and *ML pipeline* jupyter notebooks.
	
	1.4. If you would like to check the model outside of the App, run the below function in python:
	</br> filename = 'classifier.pkl'
	</br> loaded_model = pickle.load(open(filename, 'rb'))
	</br> result = loaded_model.score(X_test, y_test)
	</br> print(result)
	
2. Run the following command in the app's directory to run the *Web App*:`python run.py`. Access locally: http://localhost:3001/.

### Project Structure
Project involving building an ETL Pipeline, ML Pipeline as well as a simple Flask Web App.
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
