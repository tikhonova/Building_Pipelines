### Project Description

The goal of this project is to build a tool with easily understandable UI, which would allow people working in emergency centers quickly classify and prioritize incoming messages.

<b>Technical goals</b>: 
<ul>1. Analyze real messages that were sent during disaster events.</ul>
<ul>2. Create the best fitting machine learning model and build a pipeline for an API that classifies disaster messages. </ul>
<ul>3. Build a web interface that would allow User to input a new message and get classification results in several categories.</ul>
<ul>4. Have the web display visualizations of the training data.</ul>

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

    1.1. Navigate to the <b>Data folder</b> and run the *process_data.py* script that cleans data and stores in database: <u>`python process_data.py "messages.csv" "categories.csv" "DisasterResponse.db"`</u>
	The above command loads the datasets, fixes the data by resolving quality and tidiness issues, and loads the result into an SQLite database.
		
    1.2. <b>Navigate to the Models folder</b> and run the *train_classifier.py* script within the Model folders, which trains classifier and saves it as a pickle file: <u>`python train_classifier.py DisasterResponse.db classifier.pkl`</u>
	The second command builds and saves the model into a .pkl file. 
	
	1.3. For additional information, see the *ETL pipeline* and *ML pipeline* jupyter notebooks.
	
	1.4. If you would like to check the model outside of the App, run the below function in python:
	</br> filename = 'classifier.pkl'
	</br> loaded_model = pickle.load(open(filename, 'rb'))
	</br> result = loaded_model.score(X_test, y_test)
	</br> print(result)
	
2. <b>Navigate to the App folder</b> and run the following command in the app's directory to launch the *Web App*:<u>`python run.py`</u>. Access locally: http://localhost:3001/.

### Project Structure
Project involving building an ETL Pipeline, ML Pipeline as well as a simple Flask Web App.

<li>- app</li>
<ul>| - template</ul>
<ul>| |- master.html  # main page of web app</ul>
<ul>| |- go.html  # classification result page of web app</ul>
<ul>|- run.py  # Flask file that runs app</ul>

<li>- data</li>
<ul>|- disaster_categories.csv  # data to process </ul>
<ul>|- disaster_messages.csv  # data to process </ul>
<ul>|- process_data.py</ul>
<ul>|- InsertDatabaseName.db   # database to save clean data to </ul>

<li>- models</li>
<ul>|- train_classifier.py </ul>
<ul>|- classifier.pkl  # saved model </ul> 

<li>- README.md</li>

### App screenshots

#### <b>Main page</b>

![Screenshot_1](https://user-images.githubusercontent.com/41370639/107888575-3752d280-6edb-11eb-9de5-f759599b944e.png)

#### <b>Classifying message example -- below is the output for "There's been a storm. We need shelter."</b>
![Screenshot_2](https://user-images.githubusercontent.com/41370639/107888581-40dc3a80-6edb-11eb-8a80-ed367299f17e.png)
