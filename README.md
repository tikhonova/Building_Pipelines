### Project Description
<b>Goals</b>: 
<li>1. Analyze real messages that were sent during disaster events.</li>
<li>2. Create the best fitting machine learning model and build a pipeline for an API that classifies disaster messages. </li>
<li>3. Build a web interface that would allow User to input a new message and get classification results in several categories. </li>
<li>4. Have the web display visualizations of the training data.</li>

</br><b>Dataset</b>: imported tables with messages and given categories.

<b>Scripts</b>: 
<ul>process_data.py that cleans data and stores in database</ul>
<ul>train_classifier.py that trains and saves selected classifier</ul>
<ul>run.py that runs a web app</ul>
</br> 
<b>Additional files</b>:
<ul>SQLite database resulting from the first script</ul>
<ul>PKL file resulting from the second script</ul>
<ul>Jupyter Notebooks with exploratory analys and other work done prior to building scripts</ul>
<ul>Templates that help run the web app</ul>
</br> 

### Instructions:

1. Set up your database and model: run the following commands to load libraries and set up your database and model.

    1.1. Navigate to the <b>Data folder</b> and run the *process_data.py* script that cleans data and stores in database: <u>`python process_data.py "messages.csv" "categories.csv" "DisasterResponse.db"'</u>
	The above command loads the datasets, fixes the data by resolving quality and tidiness issues, and loads the result into an SQLite database.
		
    1.2. <b>Copy .db file into the Models folder</b>, then run the *train_classifier.py* script within the Model folders, which trains classifier and saves it as a pickle file: <u>`python train_classifier.py DisasterResponse.db classifier.pkl`</u>
	The second command builds and saves the model into a .pkl file. 
	
	1.3. For additional information, see the *ETL pipeline* and *ML pipeline* jupyter notebooks.
	
	1.4. If you would like to check the model outside of the App, run the below function in python:
	</br> filename = 'classifier.pkl'
	</br> loaded_model = pickle.load(open(filename, 'rb'))
	</br> result = loaded_model.score(X_test, y_test)
	</br> print(result)
	
2. <b>Copy .pkl and .db files into the App folder</b> and run the following command in the app's directory to launch the *Web App*:<u>`python run.py`</u>. Access locally: http://localhost:3001/.

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

### App screenshots

<b>Main page</b>
<img src="../../../../Users/piewitheye/Desktop/Screenshot_9.png" width="612" height="1176" border="0" alt="" />

<b>Classifying message example -- below is the output for "There's been a storm. We need shelter."</b>
<img src="../../../../Users/piewitheye/Desktop/Screenshot_11.png" width="610" height="1211" border="0" alt="" />
