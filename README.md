Accident Severity Prediction
This repository contains a Python script to predict the severity of accidents using a Random Forest Classifier. The classifier is trained on a dataset that contains accident data with various features such as location, nature of the accident, causes, and weather conditions.
Dependencies
•	pandas
•	scikit-learn
You can install these dependencies using pip:
pip install pandas scikit-learn
Data
The dataset, Accdataset_hk_PS_BAEL_Combined.csv, contains various features such as accident location, nature of the accident, causes, road features, intersection types, and weather conditions. The target variable is the accident severity, which is categorized into four levels: Fatal, Grevious Injury, Minor Injury, and No Injury.
Usage
1.	Clone the repository:
git clone https://github.com/username/accident-severity-prediction.git cd accident-severity-prediction 
2.	Run the script:
python accident_severity_prediction.py 
The script will train a Random Forest Classifier using the dataset, perform grid search with cross-validation to find the best hyperparameters, and make predictions on the test data. The predicted accident severities will be saved to an Excel file named predicted_output3.xlsx.
3.	Evaluate the accuracy of the model, which will be printed on the console.
Model
The model used in this script is a Random Forest Classifier from the scikit-learn library. The hyperparameters tuned during the grid search are:
•	n_estimators: The number of trees in the forest.
•	max_depth: The maximum depth of the tree.
The script uses a 5-fold cross-validation to find the best hyperparameters for the model.
Output
The script will save the predictions to an Excel file named predicted_output3.xlsx. The accuracy of the model will also be printed on the console
