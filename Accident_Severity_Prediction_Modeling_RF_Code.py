#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


# In[20]:


# Load the data
data = pd.read_csv('Accdataset_hk_PS_BAEL_Combined.csv')


# In[21]:


data.head()


# In[22]:


# Assign the target variable
y = data['Accident_Severity_C'].fillna(0)


# In[23]:


# Select feature variables
X = data.drop(['Accident_Severity_C','Accident_Index','Date','Day_of_Week','Time_of_Accident','Accident_Location_A','Accident_Location_A_Chainage_km','Accident_Location_A_Chainage_km_RoadSide','Nature_of_Accident_B1','Nature_of_Accident_B2','Nature_of_Accident_B3','Classification_of_Accident_C1','Classification_of_Accident_C2','Classification_of_Accident_C3','Causes_D1','Causes_D2','Causes_D3','Causes_D4','Causes_D5','Road_Feature_E','Road_Condition_F','Intersection_Type_G','Weather_Conditions_H','Vehicle_Type_Involved_J_V1','Vehicle_Type_Involved_J_V2','Vehicle_Type_Involved_J_V3','Vehicle_Type_Involved_J_V4','Number_of_Vehicles','Number_of_Casualties_Fatel','Number_of_Casualties_GrievousInjury','Number_of_Casualties_MinorInjury','Number_of_Casualties_NotInjured','Number_of_Casualties','Remarks'], axis=1)
X = X.fillna(0)


# In[24]:


# Mapping feature variables
accident_severity_map = {'1': 'Fatal', '2': 'Grevious Injury', '3': 'Minor Injury', '4': 'No Injury'}
data['Accident_Severity_C'] = data['Accident_Severity_C'].map(accident_severity_map)

day_of_week_map = {'1': 'Sunday', '2': 'Monday', '3': 'Tuesday', '4': 'Wednesday', '5': 'Thursday', '6': 'Friday', '7': 'Saturday'}
data['Day_of_Week'] = data['Day_of_Week'].map(day_of_week_map)


# In[25]:


# Mapping remaining feature variables
mapping = {'1': 'Urban', '2': 'Rural', '3': 'Unallocated'}
data['Accident_Location_A'] = data['Accident_Location_A'].map(mapping)

mapping = {'1': 'Overturning', '2': 'Head on collision', '3': 'Rear End Collision',
           '4': 'Collision Brush/Side Wipe', '5': 'Right Turn Collision', '6': 'Skidding',
           '7a': 'Others-Hit Cyclist', '7b': 'Others-Hit Pedestrian', '7C': 'Others-Hit Parked Vehicle',
           '7d': 'Others-Hit Fixed Object', '7e': 'Others-Wrong Side Driving', '7f': 'Others-Hit Animal',
           '7g': 'Others-Hit Two Wheeler', '7h': 'Others-Unknown', '7i': 'Others-Fallen down',
           '8': 'Overtaking vehicle', '9': 'Left Turn Collision'}

columns = ['Nature_of_Accident_B1', 'Nature_of_Accident_B2', 'Nature_of_Accident_B3']
for col in columns:
    data[col] = data[col].map(mapping)
    
mapping = {'1': 'Fatal', '2': 'Grevious Injury', '3': 'Minor Injury', '4': 'Non - Injury (Damage only)'}

columns = ['Classification_of_Accident_C1', 'Classification_of_Accident_C2', 'Classification_of_Accident_C3']
for col in columns:
    data[col] = data[col].map(mapping)

mapping = {'1': 'Drunken', '2': 'Overspeeding', '3': 'Vehicle out of control',
'4a': 'Fault of driver of motor vehicle', '4b': 'Driver of other vehicle', '4C': 'Cyclist',
'4d': 'Pedestrian', '4e': 'Passenger', '4f': 'Animal',
'5a': 'Defect in mechanical condition of motor vehicle', '5b': 'Road condition'}

columns = ['Causes_D1', 'Causes_D2', 'Causes_D3', 'Causes_D4', 'Causes_D5']
for col in columns:
    data[col] = data[col].map(mapping)
    
mapping = {'1': 'Single lane', '2': 'Two lanes', '3': 'Three lanes or more without central divider median',
           '4': 'Four lanes or more with central divider alongwith carriageway width'}
data['Road_Feature_E'] = data['Road_Feature_E'].map(mapping)

mapping = {'1': 'Straight Road', '2': 'Slight Curve', '3': 'Sharp Curve', '4': 'Flat Road', '5': 'Gentle incline',
           '6': 'Steep incline', '7': 'Hump', '8': 'Dip'}
data['Road_Condition_F'] = data['Road_Condition_F'].map(mapping)

mapping = {'1': 'T Junction', '2': 'Y Junction', '3': 'Four arm junction', '4': 'Staggered junction',
           '5': 'Roundabout', '6': 'Uncontrolled junction'}
data['Intersection_Type_G'] = data['Intersection_Type_G'].map(mapping)

mapping = {'1': 'Fine', '2': 'Mist/Fog', '3': 'Cloud', '4': 'Light Rain',
           '5': 'Heavy Rain', '6': 'Hail/sleet', '7': 'Snow', '8': 'Strong Wind', 
           '9': 'Dust Storm', '10': 'Very Hot', '11': 'Very Cold', '12': 'Other extraordinary weather condition'}
data['Weather_Conditions_H'] = data['Weather_Conditions_H'].map(mapping)

mapping = {'1': 'Car/Jeep/Van', '2': 'SUV', '3': 'Bus', '4': 'Mini Bus', '5': 'Truck', '6': 'Two Wheeler',
           '7': 'Three Wheeler', '8': 'Cycle', '9': 'Pedestrian', '10': 'Tractor', '11': 'Unknown', '12': 'Animal',
           '13': 'Objects', '14': 'LCV', '15': 'MAV'}

columns = ['Vehicle_Type_Involved_J_V1', 'Vehicle_Type_Involved_J_V2', 'Vehicle_Type_Involved_J_V3', 'Vehicle_Type_Involved_J_V4']
for col in columns:
    data[col] = data[col].map(mapping)


# In[26]:


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[27]:


# Define the model
model = RandomForestClassifier(random_state=0)


# In[28]:


# Define the hyperparameters to be tuned
param_grid = {'n_estimators': [100, 500, 1000, 5000],
              'max_depth': [2, 4, 6, 8]}


# In[29]:


# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[30]:


# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# In[31]:


results = grid_search.cv_results_
results


# In[32]:


# Use the best parameters to fit the model
clf = grid_search.best_estimator_
clf.fit(X_train, y_train)


# In[33]:


# Make predictions on the test data
import numpy 
y_pred = clf.predict(X_test)


# In[36]:


#Save the predictions as an Excel file
df = pd.DataFrame({'Predictions': y_pred})
df.to_excel('predicted_output3.xlsx', index=False)


# In[35]:


# Print the accuracy of the model
print(clf.score(X_test, y_test))


# In[ ]:




