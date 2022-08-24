# Import Libraries
import math
import os

import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima.utils import ndiffs


class PredictionModels:
    def __init__(self):
        pass

    # def plotter(self,rawDataset,trainDatset,predictedDatset):
    #     # Visualize the data
    #     plt.figure(figsize=(16, 8))
    #     plt.title(f'Model Result for {f}')
    #     plt.xlabel('Date', fontsize=18)
    #     plt.ylabel('Close Price USD ($)', fontsize=18)
    #     plt.plot(train['Close'])
    #     plt.plot(valid[['Close', 'Predictions']])
    #     plt.legend(['Train', 'Actual Price', 'Predicted Price'], loc='lower right')
    #     plt.show()

    def lstm(self):

        # open datasets file
        for f in os.listdir('Dataset'):
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.drop('Currency', axis=1)

                print(cryptoData.shape)

                # Visualize the closing price history
                plt.figure(figsize=(16, 8))
                plt.title(f'Closing Price History {f}')
                plt.plot(cryptoData['Close'])
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price USD ($)', fontsize=18)
                plt.show()

                # Create a new dataframe with only the 'Close' column
                data = cryptoData.filter(['Close'])
                # Converting the dataframe to a numpy array
                closing_price_dataset = data.values

                # Get /Compute the number of rows to train the model on
                training_data_close_price_len = math.ceil(len(closing_price_dataset) * .8)
                print(training_data_close_price_len)

                # Scale the all of the data to be values between 0 and 1
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_closing_price_data = scaler.fit_transform(closing_price_dataset)
                print(scaled_closing_price_data)
                print()
                print(0, len(scaled_closing_price_data))

                # Create the scaled training data set
                train_closing_price_data = scaled_closing_price_data[0:training_data_close_price_len, :]
                # Split the data into x_train and y_train data sets
                x_train = []
                y_train = []
                for i in range(60, len(train_closing_price_data)):
                    x_train.append(train_closing_price_data[i - 60:i, 0])
                    y_train.append(train_closing_price_data[i, 0])
                    # if i<=61:
                    #   print(x_train)
                    #   print(y_train)
                    #   print()

                # Convert x_train and y_train to numpy arrays
                x_train, y_train = np.array(x_train), np.array(y_train)
                # print(y_train)
                # print(y_train.shape)

                # Reshape the data into the shape accepted by the LSTM because it is 2D and we must reshape it 3D
                # We input the (number_of_samples,number,of,timestamps,number_of_features)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))

                # #Build the LSTM network model
                model = Sequential()
                # input_shape contains number of timestamps and number of features
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dense(units=25))
                model.add(Dense(units=1))
                # print("ok")

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                model.fit(x_train, y_train, batch_size=1, epochs=1)

                # Test data set
                test_closing_price_data = scaled_closing_price_data[training_data_close_price_len - 60:, :]
                print(1, len(test_closing_price_data))

                # Create the x_test and y_test data sets
                x_test = []
                y_test = closing_price_dataset[training_data_close_price_len:,
                         :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2032 - 1972 = 60 rows of data
                for i in range(60, len(test_closing_price_data)):
                    x_test.append(test_closing_price_data[i - 60:i, 0])

                # Convert x_test to a numpy array
                x_test = np.array(x_test)
                print(2, len(x_test))

                # Reshape the data into the shape accepted by the LSTM
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                # Getting the models predicted price values
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)  # Undo scaling

                # Calculate/Get the value of RMSE
                rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

                # Plot/Create the data for the graph
                train = data[:training_data_close_price_len]
                valid = data[training_data_close_price_len:]
                valid['Predictions'] = predictions
                # Visualize the data
                plt.figure(figsize=(16, 8))
                plt.title(f'Model Result for {f}')
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price USD ($)', fontsize=18)
                plt.plot(train['Close'])
                plt.plot(valid[['Close', 'Predictions']])
                plt.legend(['Train', 'Actual Price', 'Predicted Price'], loc='lower right')
                plt.show()

                print(rmse)

    def arima(self):
        # open datasets file
        for f in os.listdir('Dataset'):
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.drop('Currency', axis=1)

                print(cryptoData.shape)

                # Visualize the closing price history
                plt.figure(figsize=(16, 8))
                plt.title(f'Closing Price History {f}')
                plt.plot(cryptoData['Close'])
                plt.xlabel('Date', fontsize=18)
                plt.ylabel('Close Price USD ($)', fontsize=18)
                plt.show()

                # Create a new dataframe with only the 'Close' column
                data = cryptoData.filter(['Close'])
                # Converting the dataframe to a numpy array
                closing_price_dataset = data.values

                # Check if the dataset is stationary
                result_fuller_test = adfuller(closing_price_dataset)

                if result_fuller_test[1] > 0.05:
                    # Dataset is not stationary

                    # Making Series Stationary
                    closing_price_dataset_log = np.log(closing_price_dataset)

                    param = \
                        str(pm.auto_arima(closing_price_dataset_log, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = int(param[0])

                    # Finding out the 'q' parameter
                    q_param = int(param[2])

                    # Finding out the 'd' parameter
                    d_param = int(ndiffs(closing_price_dataset_log, test="adf"))

                    # Make a training dataset
                    n = int(len(closing_price_dataset_log) * 0.8)
                    train_dataset = closing_price_dataset_log[:n]
                    test_dataset = closing_price_dataset_log[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = len(test_dataset)
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(10, 8))
                    plt.plot(np.exp(train_dataset), label='Train')
                    plt.plot(np.exp(test_dataset), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

                    # result_arima_model.plot_diagnostics()

                    # print(result_arima_model.summary())

                else:
                    param = \
                        str(pm.auto_arima(closing_price_dataset, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = int(param[0])

                    # Finding out the 'q' parameter
                    q_param = int(param[2])

                    # Finding out the 'd' parameter
                    d_param = int(ndiffs(closing_price_dataset_log, test="adf"))

                    # Make a training dataset
                    n = int(len(closing_price_dataset) * 0.8)
                    train_dataset = closing_price_dataset[:n]
                    test_dataset = closing_price_dataset[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

    def ar(self):
        # open datasets file
        for f in os.listdir('Dataset'):
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.drop('Currency', axis=1)

                print(cryptoData.shape)

                # Visualize the closing price history
                # plt.figure(figsize=(16, 8))
                # plt.title(f'Closing Price History {f}')
                # plt.plot(cryptoData['Close'])
                # plt.xlabel('Date', fontsize=18)
                # plt.ylabel('Close Price USD ($)', fontsize=18)
                # plt.show()

                # Create a new dataframe with only the 'Close' column
                data = cryptoData.filter(['Close'])
                # Converting the dataframe to a numpy array
                closing_price_dataset = data.values

                # Check if the dataset is stationary
                result_fuller_test = adfuller(closing_price_dataset)

                if result_fuller_test[1] > 0.05:

                    # Dataset is not stationary

                    # Making Series Stationary
                    closing_price_dataset_log = np.log(closing_price_dataset)

                    param = \
                        str(pm.auto_arima(closing_price_dataset_log, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = int(param[0])

                    # Finding out the 'q' parameter
                    q_param = 0

                    # Finding out the 'd' parameter
                    d_param = 0

                    # Make a training dataset
                    n = int(len(closing_price_dataset_log) * 0.8)
                    train_dataset = closing_price_dataset_log[:n]
                    test_dataset = closing_price_dataset_log[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

                    # result_arima_model.plot_diagnostics()

                    # print(result_arima_model.summary())

                else:
                    param = \
                        str(pm.auto_arima(closing_price_dataset, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = int(param[0])

                    # Finding out the 'q' parameter
                    q_param = 0

                    # Finding out the 'd' parameter
                    d_param = 0

                    # Make a training dataset
                    n = int(len(closing_price_dataset) * 0.8)
                    train_dataset = closing_price_dataset[:n]
                    test_dataset = closing_price_dataset[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

    def ma(self):
        # open datasets file
        for f in os.listdir('Dataset'):
            if f.endswith(".csv"):
                cryptoData = pd.read_csv(f'Dataset/{f}', index_col=[0], parse_dates=[0], squeeze=True)
                cryptoData = cryptoData.drop('Currency', axis=1)

                print(cryptoData.shape)

                # Visualize the closing price history
                # plt.figure(figsize=(16, 8))
                # plt.title(f'Closing Price History {f}')
                # plt.plot(cryptoData['Close'])
                # plt.xlabel('Date', fontsize=18)
                # plt.ylabel('Close Price USD ($)', fontsize=18)
                # plt.show()

                # Create a new dataframe with only the 'Close' column
                data = cryptoData.filter(['Close'])
                # Converting the dataframe to a numpy array
                closing_price_dataset = data.values

                # Check if the dataset is stationary
                result_fuller_test = adfuller(closing_price_dataset)

                if result_fuller_test[1] > 0.05:

                    # Dataset is not stationary

                    # Making Series Stationary
                    closing_price_dataset_log = np.log(closing_price_dataset)

                    param = \
                        str(pm.auto_arima(closing_price_dataset_log, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = 0

                    # Finding out the 'q' parameter
                    q_param = int(param[2])

                    # Finding out the 'd' parameter
                    d_param = 0

                    # Make a training dataset
                    n = int(len(closing_price_dataset_log) * 0.8)
                    train_dataset = closing_price_dataset_log[:n]
                    test_dataset = closing_price_dataset_log[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()

                    # result_arima_model.plot_diagnostics()

                    # print(result_arima_model.summary())

                else:
                    param = \
                        str(pm.auto_arima(closing_price_dataset, start_p=0, start_q=0, max_p=5, max_q=5, test="adf",
                                          seasonal=True, trace=True)).split('(')[1]
                    param = param.replace(')', "")
                    param = param.split(',')
                    print(param)

                    # Finding out the 'p' parameter
                    p_param = 0

                    # Finding out the 'q' parameter
                    q_param = int(param[2])

                    # Finding out the 'd' parameter
                    d_param = 0

                    # Make a training dataset
                    n = int(len(closing_price_dataset) * 0.8)
                    train_dataset = closing_price_dataset[:n]
                    test_dataset = closing_price_dataset[n:]

                    print(len(train_dataset), len(test_dataset))

                    # print(train_dataset)
                    # print(test_dataset)

                    model_arima = ARIMA(train_dataset, order=(p_param, d_param, q_param))
                    result_arima_model = model_arima.fit()

                    # Our predict duration
                    step = 30
                    fc = result_arima_model.forecast(step)

                    # Take it to the orginal scale
                    fc = np.exp(fc)
                    # print(fc)
                    # print(conf)

                    fc = pd.Series(fc)
                    # lower_bound=pd.Series(conf[:,0],index=test_dataset[:step].index)
                    # upper_bound=pd.Series(conf[:,1],index=test_dataset[:step].index)

                    plt.figure(figsize=(16, 8))
                    plt.plot(np.exp(test_dataset[:step]), label='Actual Price')
                    plt.plot(fc, label="Forecast Price")
                    # plt.fill_between(lower_bound.index,lower_bound,upper_bound,color='k',alpha=0.1)
                    plt.title(f'Model Result for {f}\n ARIMA({p_param},{d_param},{q_param})')
                    plt.legend(loc='upper left')

                    plt.show()



	def hmm(self):
	print("Goodman")
	def hello():
	print("Hello")	
def hello():
	print("Hello")