from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np

### Clean up data
titanic_path = 'titanic_train.csv'
titanic = pd.read_csv(titanic_path)

# Rename target class variable
titanic.rename(columns={'Survived': 'class'}, inplace=True)

# Encode Sex and Embarked columns as numerical values
titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})

# Fill empty values
titanic = titanic.fillna(-999)

# Encode levels as digits
mlb = MultiLabelBinarizer()
CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])

# Drop unused features from dataset
titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)

# Check encoding
assert (len(titanic['Cabin'].unique()) == len(mlb.classes_)), "Not Equal"

# Add encoded features to dataset for use with TPOT
titanic_new = np.hstack((titanic_new.values,CabinTrans))

# Check encoding
assert not np.isnan(titanic_new).any()

# Store class labels
titanic_class = titanic['class'].values


### Data Analysis using TPOT

# Divide data into training and validation sets
training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, verbosity=2, max_time_mins=5) # Set generations to 5 since this lil computer can't handle default of 100
tpot.fit(titanic_new[training_indices], titanic_class[training_indices])
print(tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values))
tpot.export('tpot_titanic_pipeline.py') # Save pipeline


### Make Predictions from Submission Data





