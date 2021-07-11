import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from back_end import *

header = st.beta_container()
data_preprocessing = st.beta_container()
scatter_plot = st.beta_container()
features = st.beta_container()
models_title = st.beta_container()
models_discription = st.beta_container()
models_container = st.beta_container()
prediction = st.beta_container()

#header Selection
with header:
    st.title("Workation trip price Prediction")

rad = st.sidebar.radio("Navigation",["About","Preprocessing", "Prediction"])
if rad=="About":
    st.title("A Overview of the Project.")
    st.subheader('In the todays world people want to know things before hand or they don’t want to go to actual tours and traveler and get the quotation of the trip of the trip. Instead the just prefer to input the required details and get the cost of the trip the want to go on. Well this sounds a little difficult doesn’t it. Well it is a little fuzzy to hear but not impossible to implement. Artificial Intelligence has enabled us to predict such things.\
                Thought our application we are trying to implement such prediction and make the user experience seamless. The data we had collected is from machinehackathon. It has 21000 rows and 8 columns. It has various columns giving information about the package type, hotels, number of meals, number of fights, start city, Itinerary, and the value which we have to predict that is per person price. The data was raw. It was not at all preprocessed. The preprocessing was done to extract the necessary information into the required format. After data preprocessing the correlation between the predictive variables and the target variable. To get the correlation various techniques were used.\
                After the correlation matrix and obtaining the input data the data was passed to the models. Four models were implemented namely Multiple Linear Regression, Polynomial Linear Regression, Random Forest Regression, eXtreme Gradient Boosting(XGBoost). Performance matrix were prepared on the test and the train data and the model which gave the least error was used in making actual prediction.\
                The front-end was implemented in StreamLit a python library which helps seamless rendering of the data science application.')

elif rad=="Preprocessing":
    #data Preprocessing
    with data_preprocessing:
        st.header("Data Preprocessing!")

        #Ploting the heatmap
        st.subheader("Correllation matix")

        option = st.selectbox("Your Selection",["None","heat_map", "scatter_plot"])
        if option=="heat_map":
            fig, ax = plt.subplots()
            sns.heatmap(heatMap(), ax=ax)
            st.write(fig)
        elif option=="scatter_plot":
            with scatter_plot:
                df_encoded_copy = get_data()
                sca_col_1, sca_col_2, sca_col_3, sca_col_4, sca_col_5 = st.beta_columns(5)
                count = 1
                columns_list = ['International_Package',
                           'Package_Type_Budget', 'Package_Type_Deluxe', 'Package_Type_Luxury',
                           'Package_Type_Premium', 'Package_Type_Standard',
                           'No_Of_Destinations','Count_Of_Nights',
                           'Average_Hotel_Ratings', 'Start_City_Mumbai',
                           'Start_City_New_Delhi', 'No_Of_Airlines', 'Flights_Stops',
                           'Meals', 'No_Of_Sightseeing_Points',
                           'Rule_1', 'Rule_2', 'Rule_3']
                for i in columns_list:
                    if count>5:
                        count=1
                    else:
                        if count==1:
                            fig, ax = plt.subplots()
                            plt.scatter(list(df_encoded_copy[i]), list(df_encoded_copy['Per_Person_Price']))
                            plt.xlabel(i)
                            plt.ylabel('Per_Person_Price')
                            sca_col_1.write(fig)
                            count=count+1
                        elif count==2:
                            fig, ax = plt.subplots()
                            plt.scatter(list(df_encoded_copy[i]), list(df_encoded_copy['Per_Person_Price']))
                            plt.xlabel(i)
                            plt.ylabel('Per_Person_Price')
                            sca_col_2.write(fig)
                            count=count+1
                        elif count==3:
                            fig, ax = plt.subplots()
                            plt.scatter(list(df_encoded_copy[i]), list(df_encoded_copy['Per_Person_Price']))
                            plt.xlabel(i)
                            plt.ylabel('Per_Person_Price')
                            sca_col_3.write(fig)
                            count=count+1
                        elif count==4:
                            fig, ax = plt.subplots()
                            plt.scatter(list(df_encoded_copy[i]), list(df_encoded_copy['Per_Person_Price']))
                            plt.xlabel(i)
                            plt.ylabel('Per_Person_Price')
                            sca_col_4.write(fig)
                            count=count+1
                        elif count==5:
                            fig, ax = plt.subplots()
                            plt.scatter(list(df_encoded_copy[i]), list(df_encoded_copy['Per_Person_Price']))
                            plt.xlabel(i)
                            plt.ylabel('Per_Person_Price')
                            sca_col_5.write(fig)
                            count=count+1
        else:
            pass

    code = '''with features:
        st.header("Features Extracted")
        st.subheader("Let's take a closer look at data")
        df_table = get_data()
        fig = go.Figure(data = go.Table(
        header=dict(values=list(df_table.columns),
                   fill_color="#cc5a52",
                   align="center"),
        cells=dict(values=[df_table.International_Package,
                   df_table.Package_Type_Budget, df_table.Package_Type_Deluxe, df_table.Package_Type_Luxury,
                   df_table.Package_Type_Premium, df_table.Package_Type_Standard,
                   df_table.No_Of_Destinations,df_table.Count_Of_Nights,
                   df_table.Average_Hotel_Ratings, df_table.Start_City_Mumbai,
                   df_table.Start_City_New_Delhi, df_table.No_Of_Airlines, df_table.Flights_Stops,
                   df_table.Meals, df_table.No_Of_Sightseeing_Points,
                   df_table.Rule_1, df_table.Rule_2, df_table.Rule_3],
                   fill_color="#e3d4d3",
                   align="left")))
        fig.update_layout(margin=dict(l=5,r=5,b=10,t=10))
        st.write(fig)'''

    #diplaying encoded data
    with features:
        st.header("Features Extracted")
        st.subheader("Let's take a closer look at data")
        show = st.checkbox("Show Features")
        if show:
            st.write(get_data().head(10))

    with models_title:
        st.title("Moving on to the main event...")

        with models_discription:
            col1, col2, col3 = st.beta_columns([4,6,1])
            col1.write("")
            col2.header("Select Model")
            col3.write("")
            model = st.selectbox("", ["None","Multiple Linear Regressor","Polynomial Regressor", "Random Forest Regressor"])

    with models_container:

        train_col, test_col = st.beta_columns(2)

        if model=="Multiple Linear Regressor":
            #calling MultipleLinearRegression function
            lr_data = MultipleLinearRegression()

            train_col.subheader("Score")
            train_col.success(lr_data[0])
            train_col.subheader("Train mse")
            train_col.success(lr_data[5])
            train_col.subheader("Train mae")
            train_col.success(lr_data[7])

            test_col.subheader("Score")
            test_col.success(lr_data[1])
            test_col.subheader("Test mse")
            test_col.success(lr_data[4])
            test_col.subheader("Test mae")
            test_col.success(lr_data[6])

        elif model=="Polynomial Regressor":
            #calling Polynomial Regression function
            ply_data = PolynomialRegression()

            train_col.subheader("Score")
            train_col.success(ply_data[0])
            train_col.subheader("Train mse")
            train_col.success(ply_data[3])
            train_col.subheader("Train mae")
            train_col.success(ply_data[5])


            test_col.subheader("Score")
            test_col.success(ply_data[1])
            test_col.subheader("Test mse")
            test_col.success(ply_data[2])
            test_col.subheader("Test mae")
            test_col.success(ply_data[4])

        elif model=="Random Forest Regressor":
            #calling Polynomial Regression function
            rf_data = RandomForestRegressorModel()

            train_col.subheader("Score")
            train_col.success(rf_data[0])
            train_col.subheader("Train mse")
            train_col.success(rf_data[3])
            train_col.subheader("Train mae")
            train_col.success(rf_data[5])


            test_col.subheader("Score")
            test_col.success(rf_data[1])
            test_col.subheader("Test mse")
            test_col.success(rf_data[2])
            test_col.subheader("Test mae")
            test_col.success(rf_data[4])

else:
    with prediction:

        st.title("Let's Do Some Predictions")

        show = st.checkbox("Want to do some predictions?")
        if show:
            p_col1, p_col2, p_col3 = st.beta_columns(3)
            hyperparameters = [[]]

            is_national = p_col1.selectbox("National or Internation", ["International Package","National Package"])
            if is_national == "International Package":
                hyperparameters[0].insert(0,1)
            else:
                hyperparameters[0].insert(0,0)

            package = p_col2.selectbox("Select Package",["Budget","Deluxe","Luxury","Premium","Standard"])
            if package == "Budget":
                hyperparameters[0].insert(1,1)
                hyperparameters[0].insert(2,0)
                hyperparameters[0].insert(3,0)
                hyperparameters[0].insert(4,0)
                hyperparameters[0].insert(5,0)
            elif package == "Deluxe":
                hyperparameters[0].insert(1,0)
                hyperparameters[0].insert(2,1)
                hyperparameters[0].insert(3,0)
                hyperparameters[0].insert(4,0)
                hyperparameters[0].insert(5,0)
            elif package == "Luxury":
                hyperparameters[0].insert(1,0)
                hyperparameters[0].insert(2,0)
                hyperparameters[0].insert(3,1)
                hyperparameters[0].insert(4,0)
                hyperparameters[0].insert(5,0)
            elif package == "Premium":
                hyperparameters[0].insert(1,0)
                hyperparameters[0].insert(2,0)
                hyperparameters[0].insert(3,0)
                hyperparameters[0].insert(4,1)
                hyperparameters[0].insert(5,0)
            else:
                hyperparameters[0].insert(1,0)
                hyperparameters[0].insert(2,0)
                hyperparameters[0].insert(3,0)
                hyperparameters[0].insert(4,0)
                hyperparameters[0].insert(5,1)

            no_of_destinations = p_col3.selectbox("Select number of destination", [1,2,3,4,5,6,7,8,9,10,11])
            hyperparameters[0].insert(6,no_of_destinations)

            count_of_Nights = p_col1.selectbox("Count Of Nights", [1,2,3,4,5,6,7,8,9,10,11,12,13,14])
            hyperparameters[0].insert(7,count_of_Nights)

            date = p_col2.date_input("Date")
            year = int(date.strftime("%Y"))
            hyperparameters[0].insert(8,year)
            month = int(date.strftime("%m"))
            hyperparameters[0].insert(9,month)
            day = int(date.strftime("%d"))
            hyperparameters[0].insert(10,day)
            #print(year)

            hotel_rating = p_col3.selectbox("Hotel Ratings", [1,2,3,4,5])
            hyperparameters[0].insert(11,hotel_rating)

            start_city = p_col1.selectbox("Start City", ["Mumbai", "Delhi"])
            if start_city == "Mumbai":
                hyperparameters[0].insert(12,1)
                hyperparameters[0].insert(13,0)
            else:
                hyperparameters[0].insert(12,0)
                hyperparameters[0].insert(13,1)


            number_of_airline = p_col2.selectbox("Number of Airline", [1,2,3,4,5,6,7,8,9,10,11])
            hyperparameters[0].insert(14,number_of_airline)

            flights_stops = p_col3.selectbox("Flights Stops",[0,1,2])
            hyperparameters[0].insert(15,flights_stops)

            number_of_meals = p_col1.selectbox("Number of Meals", [2,3,4,5])
            hyperparameters[0].insert(16,number_of_meals)

            number_of_sightseeing = p_col2.selectbox("Number Of Sightseeing Points", [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
            hyperparameters[0].insert(17,number_of_sightseeing)

            hyperparameters[0].insert(18,0)
            hyperparameters[0].insert(19,1)
            hyperparameters[0].insert(20,0)
            #hyperparameters[0].insert(21,0)

            #print(hyperparameters[0])
            rf_predict = RandomForestRegressorModel()
            predicted_price = rf_predict[6].predict(hyperparameters)[0]

            st.header("Predicted Price for the Workation trip:")
            st.success("%.2f INR" % predicted_price)
