The AI-powered House Price Predictor project aims to create a strong machine-learning model that can accurately predict house prices based on factors such as location, size, number of rooms, amenities, and economic indicators. This research seeks to provide sellers, buyers, and real estate agents with useful insights into property valuation by utilizing sophisticated predictive algorithms and historical housing data.
Data Collection: The data used for training this model was gathered from Kaggle, which can be found at: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset. The dataset contained data on the price, area, number of floors, number of parking spaces, number of bedrooms and bathrooms, whether a heating system exists, whether an air conditioning unit exists, whether a basement, whether the house is near a main road, whether the house has a guest room, whether the house is in the buyer's preferred area and the furnishing status of the house.

Processing Data: The data was acquired with outliers. Therefore outliers were managed through the use of the inter-quartile range.

Model Development: A hyperparameter grid was created with an array of values for the number of estimators and the maximum depth. GridSearchCV was used to find the best possible combination and make a Random Forest regressor from these parameters.

Evaluation and Training: train_test_split was used to split the data into training and testing sets, with a size of ten percent for the testing set. The training data was used to train the model, and R-squared was used to evaluate the model's predictions of the test data. The model was fine-tuned to increase reliability, while the code was kept short and simple.

A user-friendly interface was also developed to allow the user to input their answer to various questions about the features of the house. These answers would be taken as input for the model to predict an estimated price. This value would be given as output to the user.
