import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
import random
from pypdf import PdfReader
# Define what file to read
task_to_do = {'company_name': False, 'select_company': {'total': True, 'cluster': False, 'training': False, 'predict': False, 'extend': False}, 'data_processing': {'total': True, 'cluster': False, 'training': False, 'predict': False, 'extend': False}, 'integrate': {'total': True, 'training': False, 'predict': False, 'extend': False}}

#This is setup data for company symbol reading
company_symbol_file = 'saved_file/company_symbol.pkl'
company_symbol_pdf_file = 'ru3000_membershiplist_20220624.pdf'
remove_elements = ['Ticker', 'Russell US Indexes']
remove_elements.extend([str(i) for i in range(1, 33)])
company_symbol = []


#This is setup data for company selecting data collection
start_date_cluster='2017-01-01'
end_date_cluster='2022-03-31'

start_date_training='2019-07-01'
end_date_training='2021-12-31'

start_date_predict='2022-01-03'
end_date_predict='2022-03-31'

extended_start_date_predict='2022-04-1'
extended_end_date_predict='2022-06-30'

selected_company_training_file = 'saved_file/selected_company_training.pkl'
selected_company_pre_file = 'saved_file/selected_company_predict.pkl'
selected_company_cluster_file = 'saved_file/selected_company_cluster.pkl'
selected_company_cluster_extend_file = 'saved_file/selected_company_cluster_extend.pkl'

#This is setup data for data processing
splits = 5
processed_data_training_file = 'saved_file/processed_data_training.pkl'
processed_data_pre_file = 'saved_file/processed_data_predict.pkl'
processed_data_cluster_file = 'saved_file/processed_data_cluster.pkl'
process_data_cluster_extend_file = 'saved_file/processed_data_cluster_extend.pkl'

#This is setup data for data integration
integrated_data_training_file = 'saved_file/integrated_data_training.pkl'
integrated_data_pre_file = 'saved_file/integrated_data_predict.pkl'
integrated_data_extend_file = 'saved_file/integrated_data_extend.pkl'



#Company symbol reading function
def get_company_name_from_pdf(company_symbol_pdf = company_symbol_pdf_file, company_symbol_file = company_symbol_file):
    #Read the pdf and cal the number of pages to read  
    pdf_reader = PdfReader(company_symbol_pdf)
    num_page = len(pdf_reader.pages) - 1
    
    #Read the pdf and extract the company symbol following a format
    for i in range(num_page):
        with open('russell3000.txt', 'w') as f:
            f.write(pdf_reader.pages[i].extract_text())
        with open('russell3000.txt', 'r') as f:
            lines = f.readlines()
            lines = lines[3::2]
            lines = [line.strip('\n') for line in lines]
            lines = [line for line in lines if line not in remove_elements]
            company_symbol.extend(lines)

    #Save the company symbol to a file
    with open(company_symbol_file, 'wb') as f:
        pickle.dump(company_symbol, f)
        
    if os.path.exists('russell3000.txt'):
        os.remove('russell3000.txt')

#Company selecting function
def get_price_for_selected_company(company_symbol_file_arg, start_date, end_date, selected_company_file):
    #Load company symbol from file
    with open(company_symbol_file_arg, 'rb') as f:
        company_symbol = pickle.load(f)

    cleaned_ticker = {}
    if company_symbol_file_arg != company_symbol_file:
        company_symbol = company_symbol.keys()

    #Select valid company symbol and download data from yahoo finance (only 300 companies)
    for ticker in company_symbol:   
        data = yf.download(ticker, start=start_date, end=end_date)
        if company_symbol_file_arg == company_symbol_file:
            if data.shape[0] != 1320:
                continue
        cleaned_ticker[ticker] = data
        print(len(cleaned_ticker), ticker)
        if len(cleaned_ticker) >= 300:
            break

    #Save the selected company to a file
    with open(selected_company_file, 'wb') as f:
        pickle.dump(cleaned_ticker, f) 

#Data processing function
def data_smoothing_processing(selected_company_file, processed_data_file, splits):
    #Load selected company from file
    with open(selected_company_file, 'rb') as f:
        selected_company = pickle.load(f)

    #Take the average of the data in 5-day period
    for key, value in selected_company.items():
        processed_df = pd.DataFrame()
        columns = value.columns.tolist()
        rows = value.shape[0]
        for column in columns:
            processed_df[column] = [sum(value[i:i+splits][column]) / splits for i in range(len(value) - splits + 1)]
            
        processed_df.index = value.index[4:]
        selected_company[key] = processed_df
    
    #Save the processed data to a file
    with open(processed_data_file, 'wb') as f:
        pickle.dump(selected_company, f)

#Integrate data function
def integrate_company_data(processed_data_file, integrated_data_file):
    with open(processed_data_file, 'rb') as f:
        processed_company = pickle.load(f)
    data = []
    columns = []
    index = []
    
    
    for key, item in processed_company.items():
        data.append(item['Close'].values.tolist())   
        columns.append(key)
        
    for key, item in processed_company.items():
        index = item.index
        break
    
    data = np.array(data)
    data = data.T
    
    df = pd.DataFrame(data, columns=columns, index=index)
    with open(integrated_data_file, 'wb') as f:
        pickle.dump(df, f)



#Company symbol reading process
if task_to_do['company_name']:
    get_company_name_from_pdf(company_symbol_pdf_file, company_symbol_file)


#Company selecting process
if task_to_do['select_company']['total']:
    arguments = []
    if task_to_do['select_company']['cluster']:
        arguments.append([company_symbol_file, start_date_cluster, end_date_cluster, selected_company_cluster_file])
    if task_to_do['select_company']['training']:
        arguments.append([selected_company_cluster_file, start_date_training, end_date_training, selected_company_training_file])
    if task_to_do['select_company']['predict']:
        arguments.append([selected_company_cluster_file, start_date_predict, end_date_predict, selected_company_pre_file])
    if task_to_do['select_company']['extend']:
        arguments.append([selected_company_cluster_file, extended_start_date_predict, extended_end_date_predict, selected_company_cluster_extend_file])
    for arg in arguments:
        get_price_for_selected_company(arg[0], arg[1], arg[2], arg[3])
   
        
#Data processing process
if task_to_do['data_processing']['total']:
    arguments = []
    if task_to_do['data_processing']['cluster']:
        arguments.append([selected_company_cluster_file, processed_data_cluster_file, splits])
    if task_to_do['data_processing']['training']:
        arguments.append([selected_company_training_file, processed_data_training_file, splits])
    if task_to_do['data_processing']['predict']:
        arguments.append([selected_company_pre_file, processed_data_pre_file, splits])
    if task_to_do['data_processing']['extend']:
        arguments.append([selected_company_cluster_extend_file, process_data_cluster_extend_file, splits])
    for arg in arguments:
        data_smoothing_processing(arg[0], arg[1], arg[2])

        
#Data integrating process
if task_to_do['integrate']['total']:
    arguments = []
    if task_to_do['integrate']['training']:
        arguments.append([processed_data_training_file, integrated_data_training_file])
    if task_to_do['integrate']['predict']:
        arguments.append([processed_data_pre_file, integrated_data_pre_file])
    if task_to_do['integrate']['extend']:
        arguments.append([process_data_cluster_extend_file, integrated_data_extend_file])
    for arg in arguments:
        integrate_company_data(arg[0], arg[1])
    





with open(integrated_data_pre_file, 'rb') as f:
    price_data = pickle.load(f)
    company_list = list(price_data.keys())
print(price_data[company_list[0]].shape)

with open(integrated_data_extend_file, 'rb') as f:
    price_data = pickle.load(f)
    company_list = list(price_data.keys())
print(price_data[company_list[0]].shape)