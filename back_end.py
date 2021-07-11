import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


@st.cache(suppress_st_warning=True)
def get_data():
    df = pd.read_csv("encoded_training.csv")
    return df

def heatMap():
    corrMatrix = get_data().corr()
    return corrMatrix

def MultipleLinearRegression():
    export_data=[]
    lin_df = get_data().copy()
    lin_df.dropna(inplace=True)
    lin_df['Average_Hotel_Ratings'] = lin_df['Average_Hotel_Ratings'].replace(to_replace=[0.000000], value=4.0)
    lin_df['Rule_1'] = lin_df['Rule_1'] + lin_df['Rule_NA']
    lin_df.drop(['Rule_NA'], axis=1, inplace=True)
    lin_df.drop(['Uniq_Id'], axis=1, inplace=True)
    lin_df.drop(['Day', 'Month', 'Year', 'Start_City_New_Delhi'], axis=1, inplace=True)

    training_response = lin_df.iloc[:, -1].values
    training_predictors = lin_df.iloc[:, :-1].values

    #Spliting training and testing data
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(training_predictors, training_response, test_size=0.4, random_state=1)

    #Standardizing data
    scaler = StandardScaler()
    scaled_training_predictors = scaler.fit_transform(X_Train)
    scaled_testing_predictors = scaler.fit_transform(X_Test)

    lr = LinearRegression()
    lr.fit(scaled_training_predictors, Y_Train)

    #print(lr.intercept_)
    #print(lr.score(scaled_training_predictors, Y_Train))
    lr_score_train = lr.score(scaled_training_predictors, Y_Train)
    export_data.append(lr_score_train)
    lr_score_test = lr.score(scaled_testing_predictors, Y_Test)
    export_data.append(lr_score_test)

    #Making Prediction
    Y_test_pred = lr.predict(scaled_testing_predictors)
    export_data.append(Y_test_pred)
    Y_train_pred = lr.predict(scaled_training_predictors)
    export_data.append(Y_train_pred)
    X_train_pred = lr.predict(X_Train)

    #mean_squared_error and mean_absolute_error
    test_mse = mean_squared_error(Y_Test, Y_test_pred)
    export_data.append(test_mse)
    train_mse = mean_squared_error(Y_Train, Y_train_pred)
    export_data.append(train_mse)
    test_mae = mean_absolute_error(Y_Test, Y_test_pred)
    export_data.append(test_mae)
    train_mae = mean_absolute_error(Y_Train, Y_train_pred)
    export_data.append(train_mae)
    #print(test_mse)
    #print(train_mse)
    #print(test_mae)
    #print(train_mae)

    export_data.append(X_Train)
    export_data.append(X_Test)
    export_data.append(Y_Train)
    export_data.append(Y_Test)
    #export_data.append(lr)
    export_data.append(X_train_pred)

    return export_data
    #                            0              1           2               3         4          5       6           7
    #export_data.append(lr_score_train, lr_score_test, Y_test_pred, Y_train_pred, test_mse, train_mse, test_mae, train_mae,
    #                            8        9      10        11      12
    #                        X_Train, X_Test, Y_Train, Y_Test, X_train_pred)

#MultipleLinearRegression()

def PolynomialRegression():
    export_data = []
    df_encoded_poly_copy = get_data().copy()

    df_encoded_poly_copy.dropna(inplace=True)
    df_encoded_poly_copy.drop(['Rule_NA'], axis=1, inplace=True)
    df_encoded_poly_copy.drop(['Uniq_Id'], axis=1, inplace=True)

    polynomial_features = PolynomialFeatures(degree=2)
    poly_training_predictors = polynomial_features.fit_transform(df_encoded_poly_copy.iloc[:, :-1].values)

    df_encoded_poly_copy.drop(['Package_Type_Budget', 'Start_City_Mumbai'], axis=1, inplace=True)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(poly_training_predictors, df_encoded_poly_copy.iloc[:, -1].values, test_size=0.3, random_state=1)

    scaler = StandardScaler()
    scaled_training_predictors = scaler.fit_transform(X_Train)
    scaled_testing_predictors = scaler.fit_transform(X_Test)

    lr = LinearRegression()
    lr.fit(scaled_training_predictors, Y_Train)

    export_data.append(lr.score(scaled_training_predictors, Y_Train))
    export_data.append(lr.score(scaled_testing_predictors, Y_Test))

    Y_test_pred = lr.predict(scaled_testing_predictors)
    Y_train_pred = lr.predict(scaled_training_predictors)

    test_mse = mean_squared_error(Y_Test, Y_test_pred)
    export_data.append(test_mse)
    train_mse = mean_squared_error(Y_Train, Y_train_pred)
    export_data.append(train_mse)
    test_mae = mean_absolute_error(Y_Test, Y_test_pred)
    export_data.append(test_mae)
    train_mae = mean_absolute_error(Y_Train, Y_train_pred)
    export_data.append(train_mae)

    return export_data
    #                            0              1           2         3         4          5
    #export_data.append(lr_score_train, lr_score_test, test_mse, train_mse, test_mae, train_mae)

#PolynomialRegression()

def RandomForestRegressorModel():
    export_data = []
    random_forest_df = get_data().copy()

    random_forest_df.dropna(inplace=True)
    random_forest_df.drop(['Rule_NA'], axis=1, inplace=True)
    random_forest_df.drop(['Uniq_Id'], axis=1, inplace=True)

    training_response = random_forest_df.iloc[:, -1].values
    training_predictors = random_forest_df.iloc[:, :-1].values

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(training_predictors, training_response, test_size=0.3, random_state=1)

    rf = RandomForestRegressor(random_state=42)

    rfModel = rf.fit(X_Train, Y_Train)

    export_data.append(rf.score(X_Train, Y_Train))
    export_data.append(rf.score(X_Test, Y_Test))

    Y_test_pred = rfModel.predict(X_Test)
    Y_train_pred = rfModel.predict(X_Train)

    test_mse = mean_squared_error(Y_Test, Y_test_pred)
    export_data.append(test_mse)
    train_mse = mean_squared_error(Y_Train, Y_train_pred)
    export_data.append(train_mse)
    test_mae = mean_absolute_error(Y_Test, Y_test_pred)
    export_data.append(test_mae)
    train_mae = mean_absolute_error(Y_Train, Y_train_pred)
    export_data.append(train_mae)

    export_data.append(rfModel)

    #print(rfModel.predict([[0,0,0,0,0,1,4,6,30,7,2021,0,1,0,0,2,3,0,1,0,0]]))
    return export_data

#RandomForestRegressorModel()
