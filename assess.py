import pickle
import numpy as np 
import pandas as pd

#Get price data
price_data_file = "saved_file/processed_data_predict.pkl"

#Get predict data 
predict_no_cluster_file = "saved_file/predict_no_cluster.pkl"
predict_with_cluster_file = "saved_file/predict_with_cluster.pkl"
predict_with_cluster_8_file = "saved_file/predict_with_cluster_8.pkl"
predict_with_cluster_3_file = "saved_file/predict_with_cluster_3.pkl"
predict_with_cluster_7_file = "saved_file/predict_with_cluster_7.pkl"
predict_with_all_company_file = "saved_file/predict_with_all_company.pkl"
final_day_of_training_file = "saved_file/integrated_data_training.pkl"

with open(price_data_file, 'rb') as f:
    price_data = pickle.load(f)
    company_list = list(price_data.keys())

with open(predict_no_cluster_file, 'rb') as f:
    predict_no_cluster = pickle.load(f)
    
with open(predict_with_cluster_file, 'rb') as f:
    predict_with_cluster = pickle.load(f)
    #predict_with_cluster = pd.DataFrame(predict_with_cluster, columns=company_list)
    
with open(predict_with_all_company_file, 'rb') as f:
    predict_with_all_company = pickle.load(f)
    #predict_with_all_company = pd.DataFrame(predict_with_all_company, columns=company_list)

with open(predict_with_cluster_8_file, 'rb') as f:
    predict_with_cluster_8 = pickle.load(f)
    #predict_with_all_company = pd.DataFrame(predict_with_all_company, columns=company_list)
    
with open(predict_with_cluster_3_file, 'rb') as f:
    predict_with_cluster_3 = pickle.load(f)
    #predict_with_all_company = pd.DataFrame(predict_with_all_company, columns=company_list)
    
with open(predict_with_cluster_7_file, 'rb') as f:
    predict_with_cluster_7 = pickle.load(f)
    #predict_with_all_company = pd.DataFrame(predict_with_all_company, columns=company_list)

with open(final_day_of_training_file, 'rb') as f:
    final_day_of_training = pickle.load(f)
    final_day_of_training = final_day_of_training.tail(1)

pd.options.mode.chained_assignment = None


def buy_and_hold(price_data):
    columns = ['Company', 'Start Open', 'End Open', 'Return']
    data = []
    for company in company_list:
        start_open = price_data[company]['Open'].values[0]
        end_open = price_data[company]['Open'].values[-1]
        # return_gain = investment / start_open * end_open
        # percent_gain = (return_gain - investment) / investment * 100
        return_percent = (end_open - start_open) / start_open * 100
        data.append([company, start_open, end_open, return_percent])
    df = pd.DataFrame(data, columns=columns)
    return df, df['Return'].mean()

def perfect_forecast(price_data):
    columns = ['Company', 'Return']
    data = []
    for company in company_list:
        company_df = price_data[company]
        company_df = company_df[['Open', 'Close']]
        return_array = np.array([])
        for index, row in company_df.iterrows():
            if row['Close'] > row['Open']:
                day_return = (row['Close'] - row['Open']) / row['Open'] * 100
            elif row['Close'] < row['Open']:
                day_return = (row['Open'] - row['Close']) / row['Close'] * 100
            return_array = np.append(return_array, day_return)
        company_df['Return'] = return_array
        data.append([company, company_df['Return'].sum()])
    
    df = pd.DataFrame(data, columns=columns)
    return df, df['Return'].mean() 


def lstm_predict(price_data, predict_data):
    columns = ['Company', 'Return']
    data = []
    for company in company_list:
        predict_price = list(predict_data[company].values)
        
        company_df = price_data[company]
        company_df = company_df[['Open', 'Close']]
        
        return_array = np.array([])
        
        close_price = list(company_df['Close'].values)
        close_price.insert(0, final_day_of_training[company].values[0])
        
        open_price = company_df['Open'].values 
        
        # print(len(predict_price))
        # print(len(open_price))
        
        for i in range(len(open_price)):
            if predict_price[i] > open_price[i] and predict_price[i] > close_price[i]:
                day_return = (close_price[i+1] - open_price[i]) / open_price[i] * 100
            elif predict_price[i] < open_price[i] and predict_price[i] < close_price[i]:
                day_return = (open_price[i] - close_price[i+1]) / close_price[i+1] * 100
            return_array = np.append(return_array, day_return)
        company_df['Return'] = return_array
        data.append([company, company_df['Return'].sum()])
    df = pd.DataFrame(data, columns=columns)
    return df, df['Return'].mean()

columns = ['Method', 'Return']
data = []

buy_hold_return_df, buy_hold_return_percent = buy_and_hold(price_data)
data.append(['Buy and Hold', buy_hold_return_percent])


perfect_forecast_df, perfect_forecas_return_percent = perfect_forecast(price_data)
data.append(['Perfect Forecast', perfect_forecas_return_percent])

# lstm_with_all_company_df, lstm_with_all_company_return_percent = lstm_predict(price_data, predict_with_all_company)
# data.append(['LSTM with All Company', lstm_with_all_company_return_percent])


lstm_no_cluster_df, lstm_no_cluster_return_percent = lstm_predict(price_data, predict_no_cluster)
data.append(['LSTM with no Cluster', lstm_no_cluster_return_percent])

lstm_with_cluster_df, lstm_with_cluster_return_percent = lstm_predict(price_data, predict_with_cluster_3)
data.append(['LSTM with 3 Clusters', lstm_with_cluster_return_percent])

lstm_with_cluster_df, lstm_with_cluster_return_percent = lstm_predict(price_data, predict_with_cluster_7)
data.append(['LSTM with 7 Clusters', lstm_with_cluster_return_percent])

lstm_with_cluster_8_df, lstm_with_cluster_8_return_percent = lstm_predict(price_data, predict_with_cluster_8)
data.append(['LSTM with 8 Clusters', lstm_with_cluster_8_return_percent])



df = pd.DataFrame(data, columns=columns)
print(df)