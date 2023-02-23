# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GenerateCSV:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        
    def write(self):
        # Create a dummy csv file
        dates = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', 
                 '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01',
                 '2021-11-01', '2021-12-01', '2022-01-01', '2022-02-01', '2022-03-01', 
                 '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01',
                 '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01']
                 file_sizes =[1.5, 1.6, 1.8, 2, 2.1, 2.3, 2.4, 2.6, 2.7, 2.9, 3, 3.2,
                     3.3, 3.5, 3.6, 3.8, 3.9, 4.1, 4.2, 4.4, 4.5, 4.7, 4.8, 5]
        records = [1000, 1200, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
                   5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000,
                   9500, 10000, 10500, 11000, 11500, 12000]
        df = pd.DataFrame({'submitted_date': dates,
                           'file_size_gb': file_sizes,
                           'records': records})
        df.to_csv(self.csv_filename, index=False)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor 

class RegressionModel:
    def __init__(self, data_file):
        # Load the dataset into a pandas dataframe
        self.df = pd.read_csv(data_file)

        # Feature engineering
        self.df['submitted_date'] = pd.to_datetime(self.df['submitted_date'])
        self.df['year'] = self.df['submitted_date'].dt.year
        self.df['month'] = self.df['submitted_date'].dt.month
        
        # Split the data
        self.X = self.df[['year', 'month', 'records']]
        self.y = self.df['file_size_gb']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Choose a (Linear) regression model
        self.model = LinearRegression()
        # self.model = DecisionTreeRegressor()
        # self.model = RandomForestRegressor(n_estimators=10)
        
    def train(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, month, year, records):
        # Make a prediction for a new data point
        new_data = pd.DataFrame({'year': [year], 'month': [month], 'records': [records]})
        prediction = self.model.predict(new_data)[0]
        return prediction
    
    def evaluate(self):
        # Evaluate the model's performance
        y_pred = self.model.predict(self.X_test)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        w = np.array(self.y_test)
        wmape = np.sum(w * np.abs((self.y_test - y_pred) / self.y_test)) * 100 / np.sum(w)
        return mape, wmape


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

class ARIMAModel:
    def __init__(self, data_file):
        # Load the dataset into a pandas dataframe
        self.df = pd.read_csv(data_file)
        self.df.set_index('submitted_date', inplace=True)
        self.model = None
        
    def fit(self, p, d, q):
        # Fit the model
        self.model = ARIMA(endog=self.df['file_size_gb'], order=(p,d,q))
        self.results = self.model.fit()
            
    def forecast(self, steps=1):
        # Make a forecast
        return self.results.forecast(steps=steps)

    def stationary(self):
        # Check stationary
        result = adfuller(df['file_size_gb'])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        if result[1] > 0.05:
            print('Data is non-stationary')
        else:
            print('Data is stationary')
            
    def trend(self, freq):
        # Check seasonality
        # subtraction = self.df.index[1] - self.df.index[0]
        # if '7' in str(subtraction):
            # freq = 52
        # elif '1' in str(subtraction):
            # freq = 365        
        # Seasonal decomposition
        # set the model as additive and period as 12 months as we have monthly data
        decomposition = seasonal_decompose(self.df['file_size_gb'], period= freq)
        decomposition = seasonal_decompose(self.df['file_size_gb'], period=12, model='additive', period=12)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        # If the seasonal plot shows a clear and stable pattern over time, 
        # it indicates that the data is non-stationary. 
        # In contrast, if the seasonal plot does not show any clear pattern, 
        # it indicates that the data is stationary. 
        # If the residual plot shows a white noise pattern (random fluctuations), 
        # it suggests that the data is stationary. 
        # If there is a clear pattern in the residual plot, 
        # it suggests that the data is non-stationary.
        # Plot components, the original series, trend, seasonal, and residual components using subplots
        plt.figure(figsize=(12,8))
        plt.subplot(411)
        plt.plot(df.index, df['file_size_gb'], label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(df.index, trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(df.index, seasonal,label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(df.index, residual, label='Residuals')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
    def plot(self):
        # Perform plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        self.results.plot_predict(start=1, end=len(self.df)+12, ax=ax)
        ax.set_xlabel('submitted_date')
        ax.set_ylabel('file_size_gb')
        plt.show()


if __name__ == '__main__':
    # Create a dummy csv file
    GenerateCSV('file_size.csv').write()
    
    # Or use CSV file name simply to import 
    
    # Create an instance of the ByLinearRegression class
    predictor = RegressionModel('file_sizes.csv')

    # Train the model
    predictor.train()

    # Make a prediction for January 2023 with 1000 records uploaded
    prediction = predictor.predict(2023, 1, 1000)
    print("The predicted file size for January 2023 is:", prediction)

    # Evaluate the model's performance
    mape, wmape = predictor.evaluate()
    print(f"MAPE: {mape:.2%}, WMAPE: {wmape:.2%}")   
        
    # create an instance of the ARIMAModel class
    arima_model = ARIMAModel('file_sizes.csv')

    # fit the ARIMA model to the data
    arima_model.fit(1, 0, 0)

    # forecast the file size for the next month
    forecast = arima_model.forecast()
    print(forecast)
    
    # check data stationary
    arima_model.stationary()    
    
    # plot data with ADF test
    arima_model.plot()