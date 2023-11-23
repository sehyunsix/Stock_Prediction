
import pandas as pd
import numpy as np
import data_processing
import torch
from torch.utils.data import Dataset ,DataLoader


WINDOW_SIZE  = 90
PREDICT_SIZE = 30
SLIDING_SIZE =1
FEATURE_SIZE = 6
BATCH_SIZE =64
FULLLY_SIZE =128
FEATURE = [ 'Open', 'High','Low','Close', 'Volume','sell','buy' ]
TARGET = 'Close'
TICKER_NUMBER =2743
TOTAL_DAY = 166
WINDOW_NUMBER =int(TOTAL_DAY - (WINDOW_SIZE+PREDICT_SIZE)/SLIDING_SIZE+1)



def min_max(sequences):
  results = sequences.copy()
  v_min =results.min()
  v_max =results.max()
  new_min =0
  new_max =1
  min_max=[]
  for index,sequence in enumerate(results):
    v_min =sequence.min()
    v_max =sequence.max()
    v_p = (sequence - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    min_max.append([v_min,v_max])
    results[index] = v_p
  return results, min_max

def inverse_min_max(sequences,min_max):
  original = []
  for index,sequence in enumerate(sequences):
    v_min =min_max[index][0]
    v_max =min_max[index][1]
    sequence_restored = (sequence) * (v_max - v_min) + v_min
    original.append(sequence_restored)
  return np.array(original)

def preprocess_lstm(df):
  sequences = list()
  for group in df.groupby('ticker'):
    sequences.append(group[1][FEATURE])
  sequences=np.array(sequences)

  ## min_max trading
  price , trade,sentiments =np.split(sequences,[4,5],axis=2)
  trade , min_max_trading =min_max(trade)
  price , min_max_list =min_max(price)
  combine =np.concatenate([price,trade,sentiments],axis =2)

  ## min_max another

  ## min_max another
  result_list= []
  for i in range(0,WINDOW_NUMBER):
    a, b, c= np.split(combine,[i,i+120],axis=1)
    result_list.append(b)
  result_array = np.array(result_list)
  train ,vaild = np.split(result_array,[46],axis=0)
  train_x,train_y=np.split(train.reshape(46*TICKER_NUMBER,WINDOW_SIZE+PREDICT_SIZE,FEATURE_SIZE),[WINDOW_SIZE],axis=1)
  vaild_x,vaild_y=np.split(vaild.reshape(1*TICKER_NUMBER,WINDOW_SIZE+PREDICT_SIZE,FEATURE_SIZE),[WINDOW_SIZE],axis=1)
  return train_x,train_y,vaild_x,vaild_y,min_max_list

class BaseDataset(Dataset):
    def __init__(self, x_data, y_data, target):
        self.x_data = torch.tensor(x_data,dtype =torch.float32)
        self.y_data = torch.tensor(y_data[:,:,FEATURE.index(target)],dtype = torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]

# def make_train_loader():
#   data= pd.read_csv('stockPrice/NASDAQ.csv')
#   data.columns = ['Date', 'Ticker', 'Open', 'High','Low', 'Close', 'Volume','sell','buy' ]
#   train_x,train_y,vaild_x,vaild_y,min_max_list = preprocess_lstm(data)
#   train_data = BaseDataset(train_x,train_y,TARGET)
#   test_data =  BaseDataset(vaild_x,vaild_y,TARGET)
#   train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
#   test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
#   return train_dataloader,test_dataloader ,train_x,train_y,vaild_x,vaild_y,min_max_list
