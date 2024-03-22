#Libraries
import pandas as pd
import numpy as np
#from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
# Read the data from the dataframe
data = pd.read_csv("Housing_Dataset.csv")

#Check for missing values
#missing_val = data.isnull().sum()
#print(missing_val) <--- No missing values in any column

#Manage outliers
q1_pr=np.percentile(data['price'],25)
q2_pr=np.percentile(data['price'],75)
iqr_pr=q2_pr-q1_pr
lower_pr=q1_pr - 1.5*iqr_pr
upper_pr=q2_pr + 1.5*iqr_pr
data=data[(data['price']>=lower_pr) & (data['price']<=upper_pr)]

q1_ar=np.percentile(data['area'],25)
q2_ar=np.percentile(data['area'],75)
iqr_ar=q2_ar-q1_ar
lower_ar=q1_ar-1.5*iqr_ar
upper_ar=q2_ar+1.5*iqr_ar
data=data[(data['area']>=lower_ar) & (data['area']<=upper_ar)]


#Convert Boolean values into numerical data
data = pd.get_dummies(data, columns=['mainroad','guestroom','basement','hotwaterheating', 'airconditioning','prefarea','furnishingstatus'])

#Make a hyperparameter grid
para_grid = {
    'n_estimators': [65,75,85],
    'max_depth': [3,5,7],
}

#Define the features and label
x_features = data.drop('price', axis=1).values
y_labels = data['price'].values

#Split the features and labels into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.1, random_state = 42)

#Create the model with the best parameters
rf_regressor = RandomForestRegressor(random_state=42)

gSearch = GridSearchCV(estimator = rf_regressor, param_grid = para_grid, cv=5, scoring = 'neg_mean_squared_error')
gSearch.fit(x_train, y_train)

best_paras = gSearch.best_params_

best_rfr = RandomForestRegressor(**best_paras, random_state = 42)
best_rfr.fit(x_train, y_train)

#Predict from the testing sets and evaluate the model
#prediction = best_rfr.predict(x_test)

#Evaluation using MSE, MAE, and R-Squared
#mae = mean_absolute_error(y_test, prediction)
#mse = mean_squared_error(y_test, prediction)
#r2 = r2_score(y_test, prediction)

#Variables for the input
mainroad_yes = ''
mainroad_no = ''
guestRoom_yes = ''
guestRoom_no = ''
basement_yes = ''
hotWater_yes = ''
hotWater_no = ''
AC_yes = ''
AC_no = ''
prefarea_yes = ''
prefarea_no = ''
furnished = ''
semi_furnished = ''
unfurnished = ''
basement_no = ''
stories = ''
parking = ''
bedrooms = ''
bathrooms =  ''
area = ''

#Ask use for input
choice = ''
while choice != 'yes' or choice != 'no':
  choice = input("Would you like the house to be at the main road? Yes/No ")
  if choice.lower() == 'yes':
    mainroad_yes = 1
    mainroad_no = 0
    break
  elif choice.lower() == 'no':
    mainroad_yes = 0
    mainroad_no = 1
    break
  else:
    print("Please enter a valid answer. Yes/No")

choice = ''
while choice != 'yes' or choice != 'no':
  choice = input("Would you like the house to contain a guest room? Yes/No ")
  if choice.lower() == 'yes':
    guestRoom_yes = 1
    guestRoom_no = 0
    break
  elif choice.lower() == 'no':
    guestRoom_yes = 0
    guestRoom_no = 1
    break
  else:
    print("Please enter a valid answer. Yes/No")

choice = ''
while choice != 'yes' or choice != 'no':
  choice = input("Would you like the house to contain a basement? Yes/No ")
  if choice.lower() == 'yes':
    basement_yes = 1
    basement_no = 0
    break
  elif choice.lower() == 'no':
    basement_yes = 0
    basement_no = 1
    break
  else:
    print("Please enter a valid answer. Yes/No")

choice = ''
while choice != 'yes' or choice != 'no':
  choice = input("Would you like the house to have a heater to heat the water? Yes/No ")
  if choice.lower() == 'yes':
    hotWater_yes = 1
    hotWater_no = 0
    break
  elif choice.lower() == 'no':
    hotWater_yes = 0
    hotWater_no = 1
    break
  else:
    print("Please enter a valid answer. Yes/No")

choice = ''
while choice != 'yes' or choice != 'no':
  choice = input("Would you like the house to have an AC? Yes/No ")
  if choice.lower() == 'yes':
    AC_yes = 1
    AC_no = 0
    break
  elif choice.lower() == 'no':
    AC_yes = 0
    AC_no = 1
    break
  else:
    print("Please enter a valid answer. Yes/No")

choice = ''
while choice != 'yes' or choice != 'no':
  choice = input("Would you like the house to be in a very developed area? Yes/No ")
  if choice.lower() == 'yes':
    prefarea_yes = 1
    prefarea_no = 0
    break
  elif choice.lower() == 'no':
    prefarea_yes = 0
    prefarea_no = 1
    break
  else:
    print("Please enter a valid answer. Yes/No")

choice = ''
while choice != 'furnished' or choice != 'semi-furnished' or choice != "unfurnished":
  choice = input("Would you like the house to be furnished, semi-furnished, or unfurnished? ")
  if choice.lower() == 'furnished':
    furnished = 1
    semi_furnished = 0
    unfurnished = 0
    break
  elif choice.lower() == 'semi-furnished':
    furnished = 0
    semi_furnished = 1
    unfurnished = 0
    break
  elif choice.lower() == 'unfurnished':
    furnished = 0
    semi_furnished = 0
    unfurnished = 1
    break
  else:
    print("Please enter a valid answer.")

while True:
    stories = input("How many floors should the house have? ")
    try:
        stories = int(stories)
        break
    except ValueError:
        print("Enter an integer without decimal points.")

while True:
    parking = input("How many parking spaces should the house have? ")
    try:
        stories = int(parking)
        break
    except ValueError:
        print("Enter an integer without decimal points.")

while True:
    bedrooms = input("How mnany bedrooms do you want? ")
    try:
        stories = int(bedrooms)
        break
    except ValueError:
        print("Enter an integer without decimal points.")

while True:
    bathrooms =  input("How many bathrooms do you want? ")
    try:
        stories = int(bathrooms)
        break
    except ValueError:
        print("Enter an integer without decimal points.")

while True:
    area = input("What is the approxiamte area of the house you wish to have? ")
    try:
        stories = int(area)
        break
    except ValueError:
        print("Enter an integer without decimal points.")

#predict price using input
pred = best_rfr.predict(np.array([[area, bedrooms, bathrooms, stories, parking, mainroad_no, mainroad_yes, guestRoom_no, guestRoom_yes, basement_no, basement_yes, hotWater_no, hotWater_yes, AC_no, AC_yes, prefarea_no, prefarea_yes, furnished, semi_furnished, unfurnished]]))
pred = round(pred[0])
print("Estimated price of the house: " + str(pred))
