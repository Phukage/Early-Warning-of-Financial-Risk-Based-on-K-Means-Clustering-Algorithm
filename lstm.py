import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D

#Configure what to do
task_to_do = {'train_lstm_model': {'total': True, 'no_cluster': False, 'cluster': {'total': True, '3': False, '7': False, '8': False}, 'all_company': False, 'extend': False},
              'predict': {'total': True, 'no_cluster': False, 'cluster': {'total': True, '3': False, '7': False, '8': False}, 'all_company': False, 'extend': True}}


#Getting the cluster label
cluster_file_3 = 'output_cluster_high_sil/cluster_3_0.9403615465685364.pickle'
cluster_file_8 = 'output_cluster_high_sil/cluster_8_0.5698056783710818.pickle'
cluster_file_7 = 'output_cluster_high_sil/cluster_7_0.5696079164725919.pickle'

#Getting the dataset for training and prediction (columns: company, rows: price per day)
integrated_data_training = 'saved_file/integrated_data_training.pkl'
integrated_data_predict = 'saved_file/integrated_data_predict.pkl'
integrated_data_predict_extend = 'saved_file/integrated_data_extend.pkl'

#The saved file for predicting result
predict_no_cluster_file = "saved_file/predict_no_cluster.pkl"
predict_with_cluster_3_file = "saved_file/predict_with_cluster_3.pkl"
predict_with_cluster_7_file = "saved_file/predict_with_cluster_7.pkl"
predict_with_cluster_8_file = "saved_file/predict_with_cluster_8.pkl"
predict_with_all_company_file = "saved_file/predict_with_all_company.pkl"
predict_with_cluster_extend_file = "saved_file/predict_with_cluster_extend"

#LSTM model file
lstm_model_3_cluster_file = 'lstm_model/lstm_model_3_cluster'
lstm_model_7_cluster_file = 'lstm_model/lstm_model_7_cluster'
lstm_model_8_cluster_file = 'lstm_model/lstm_model_8_cluster'
lstm_model_no_cluster_file = 'lstm_model/lstm_model_no_cluster'
lstm_model_all_company_file = 'lstm_model/lstm_model_all_company'

def get_cluster_company(cluster_label, company_list, company):
    pos = company_list.index(company)
    cluster_group = cluster_label[pos]
    cluster_label_array = np.array(cluster_label)
    company_list_array = np.array(company_list)
    return company_list_array[cluster_label_array == cluster_group]

def train_lstm_model(training_data_file, cluster_label_file, lstm_model_file, all_flag = False):
    #Get the data set for training and predicting
    with open(training_data_file, 'rb') as f:
        cprice_data_training = pickle.load(f)
        
    if cluster_label_file != 'None':
        with open(cluster_label_file, 'rb') as f:
            cluster_label = pickle.load(f)
        
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    company_list = cprice_data_training.columns.tolist()
    
    scaled_cprice_data_training = scaler.fit_transform(cprice_data_training[company_list].values)
    scaled_cprice_df_training = pd.DataFrame(scaled_cprice_data_training, columns=company_list)
    count = 0
    for company in company_list:
        count += 1
        print(lstm_model_file, count)
        #Get the companies in the same cluster
        cluster_company = [company]
        if cluster_label_file != 'None':
            cluster_company = get_cluster_company(cluster_label, company_list, company)
        if all_flag:
            cluster_company = company_list
        n_feature = len(cluster_company)
        
        #Create the training data
        x_training = []
        y_training = []
        for i in range(5, len(scaled_cprice_df_training)):
            x_training.append(scaled_cprice_df_training[cluster_company].values[i-5:i])
            y_training.append(scaled_cprice_df_training[company].values[i])
        
        x_training = np.array(x_training)
        y_training = np.array(y_training)
        if cluster_label_file == 'None':
            x_training = np.reshape(x_training, (x_training.shape[0], x_training.shape[1], n_feature))
        
        #Create the model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_training.shape[1], n_feature)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128, activation='relu',return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, activation='relu',return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=60, activation='relu'))
        model.add(Dense(units=1))
        
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.fit(x_training, y_training, epochs=100, batch_size=32)
    
        saved_file = lstm_model_file + f"{count}.pkl"
        with open(saved_file, 'wb') as f:
            pickle.dump(model, f)

def lstm_predict(training_data_file, predict_data_file, cluster_label_file, target_file, lstm_model_file, all_flag=False):
    #Get the data set for training and predicting
    with open(training_data_file, 'rb') as f:
        cprice_data_training = pickle.load(f)
    with open(predict_data_file, 'rb') as f:
        cprice_data_predict = pickle.load(f)
    if cluster_label_file != 'None':
        with open(cluster_label_file, 'rb') as f:
            cluster_label = pickle.load(f)
        
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    company_list = cprice_data_training.columns.tolist()
    
    scaled_cprice_data_training = scaler.fit_transform(cprice_data_training[company_list].values)
    scaled_cprice_df_training = pd.DataFrame(scaled_cprice_data_training, columns=company_list)
    
    scaled_cprice_data_predict = scaler.fit_transform(cprice_data_predict[company_list].values)
    scaled_cprice_data_predict = np.append(scaled_cprice_data_training[-5:], scaled_cprice_data_predict, axis=0)
    scaled_cprice_df_predict = pd.DataFrame(scaled_cprice_data_predict, columns=company_list)
    
    

    predict_result = pd.DataFrame()
    count = 0
    df_ls = []
    for company in company_list:
        #Get the companies in the same cluster
        cluster_company = [company]
        if cluster_label_file != 'None':
            cluster_company = get_cluster_company(cluster_label, company_list, company)
        if all_flag:
            cluster_company = company_list
        n_feature = len(cluster_company)
        
        #Create the prediction data
        x_predict = []
        for i in range(5, len(scaled_cprice_df_predict)):
            x_predict.append(scaled_cprice_df_predict[cluster_company].values[i-5:i])
        
        x_predict = np.array(x_predict)
        if cluster_label_file == 'None':
            x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], n_feature))

        print(x_predict.shape)
        #Load the model
        count += 1
        print(f"Predicted {cluster_label_file} : {count} : {company} companies")
        saved_file = lstm_model_file + f"{count}.pkl"
        with open(saved_file, 'rb') as f:
            model = pickle.load(f)
        
        #Predict the data
        predictions = model.predict(x_predict)
        predictions = predictions.reshape(-1)
        predict_result = pd.DataFrame(predictions, columns=[company])
        df_ls.append(predict_result)
        
    

    predict_result = pd.concat(df_ls, axis=1)
    predict_result = scaler.inverse_transform(predict_result)
    predict_result = pd.DataFrame(predict_result, columns=company_list)
    
    with open(target_file, 'wb') as f:
        pickle.dump(predict_result, f)
        

    
if task_to_do['train_lstm_model']['total']:
    if task_to_do['train_lstm_model']['no_cluster']:
        train_lstm_model(integrated_data_training, 'None', lstm_model_no_cluster_file)
    if task_to_do['train_lstm_model']['cluster']['3']:
        train_lstm_model(integrated_data_training, cluster_file_3, lstm_model_3_cluster_file)
    if task_to_do['train_lstm_model']['cluster']['7']:
        train_lstm_model(integrated_data_training, cluster_file_7, lstm_model_7_cluster_file)
    if task_to_do['train_lstm_model']['cluster']['8']:
        train_lstm_model(integrated_data_training, cluster_file_8, lstm_model_8_cluster_file)
    if task_to_do['train_lstm_model']['all_company']:
        train_lstm_model(integrated_data_training, 'None', lstm_model_all_company_file, all_flag=True)


if task_to_do['predict']['total']:
    if task_to_do['predict']['no_cluster']:
        lstm_predict(integrated_data_training, integrated_data_predict, 'None', predict_no_cluster_file, lstm_model_no_cluster_file)
    if task_to_do['predict']['cluster']['3']:
        lstm_predict(integrated_data_training, integrated_data_predict, cluster_file_3, predict_with_cluster_3_file, lstm_model_3_cluster_file)
    if task_to_do['predict']['cluster']['7']:
        lstm_predict(integrated_data_training, integrated_data_predict, cluster_file_7, predict_with_cluster_7_file, lstm_model_7_cluster_file)
    if task_to_do['predict']['cluster']['8']:
        lstm_predict(integrated_data_training, integrated_data_predict, cluster_file_8, predict_with_cluster_8_file, lstm_model_8_cluster_file)
    if task_to_do['predict']['all_company']:
        lstm_predict(integrated_data_training, integrated_data_predict, 'None', predict_with_all_company_file, lstm_model_no_cluster_file ,all_flag=True)
    if task_to_do['predict']['extend']:
        lstm_predict(integrated_data_training, integrated_data_predict_extend, cluster_file_3, predict_with_cluster_extend_file + "3.pkl", lstm_model_3_cluster_file,all_flag=True)
        lstm_predict(integrated_data_training, integrated_data_predict_extend, cluster_file_7, predict_with_cluster_extend_file + "7.pkl", lstm_model_7_cluster_file,all_flag=True)
        lstm_predict(integrated_data_training, integrated_data_predict_extend, cluster_file_8, predict_with_cluster_extend_file + "8.pkl", lstm_model_8_cluster_file,all_flag=True)
    
    
