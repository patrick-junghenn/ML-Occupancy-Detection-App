# Occupancy Detection Application
This application utilizes machine learning to predict if a space is occupied or vacant. By taking measurements of temperature, CO2, humidity, humidity ratio, and light from the local environment, the application can make inferences as to whether a space is occupied or vacant. 

This application gives the user a variety of machine learning algorithms to choose from. Each algorithm includes the appropriate hyperparameters, which can be modified by the user.

The folder "Demo-images" contains screenshots of the application and its features.

## Application
This folder contains the codes needed to run the occupancy detection application.

### Application Requirements
To successfully run the application, use a python 3.7 interpreter.

This application requires the following libraries:

  * sys
  * pandas
  * numpy
  * sklearn
  * Matplotlib
  * PyQt5
  * pydotplus
  * xgboost
  * webbrowser
  * random
  * warnings

### Running GUI
To run this project, clone the repository and run the occupancy-detection-app.py file located in the Application folder. Be sure that occupancy-data.csv is in the Application folder as well. When the application is executed, a window describing its layout and features will be shown. All of the features are located in the menu bar, along the top of the screen.

## Tools & Features
### Analyze Distribution
This menu selection tab contains a correlation matrix that displays the distributions of occupancy detector readings. A user has the option to add and remove predictors.

### Model Selection
This menu selection tab contains the machine learning models available for use. Each model displays the output results, diagnostic plots, and any relevant hyperparameter options.

Available Models:

  * Decision Tree
  * Random Forest
  * Logistic Regression
  * K-nearest Neighbors
  * XGBoost
  * AdaBoost
  
### Predict Occupancy Status
This window allows a user to input raw data to make real-time predictions.
