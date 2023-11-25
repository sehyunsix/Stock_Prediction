
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset ,DataLoader
import pandas as pd

class Window_maker:
  def __init__(self,window_size:int,sliding_size:int,predict_size:int,data:pd.DataFrame) -> None:
    self.window_size = window_size
    self.sliding_size = sliding_size
    self.predict_size = predict_size
    self.feature =  [ 'Open', 'High','Low','Close', 'Volume']
    self.feature_size = len(self.feature)
    self.data = data
    self.ticker_number =len(data['Ticker'].unique())
    self.total_day = len(data['Date'].unique())
    self.window_number =int(self.total_day - (self.window_size+self.predict_size)/self.sliding_size+1)

  def min_max(self,sequences):
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

  def inverse_min_max(sequences, min_max):
    original = []
    for index, sequence in enumerate(sequences):
      v_min = min_max[index][0]
      v_max = min_max[index][1]
      sequence_restored = (sequence) * (v_max - v_min) + v_min
      original.append(sequence_restored)
      result = torch.stack(original)
    return result



  def make_split(self,sequences):
    original_list = []
    for i in range(0,self.window_number):
      a, b, c= np.split(sequences,[i,i+self.window_size+self.predict_size],axis=1)
      original_list.append(b)
    result_array = np.array(original_list)
    train ,vaild = np.split(result_array,[-1],axis=0)
    train_x,train_y=np.split(train.reshape((self.window_number-1)*self.ticker_number,self.window_size+self.predict_size,self.feature_size),[self.window_size],axis=1)
    vaild_x,vaild_y=np.split(vaild.reshape(1*self.ticker_number,self.window_size+self.predict_size,self.feature_size),[self.window_size],axis=1)
    return train_x,train_y,vaild_x,vaild_y


  def preprocess_window(self):

    sequences = list()
    for group in self.data.groupby('Ticker'):
      sequences.append(group[1][self.feature])
    sequences=np.array(sequences)
    original = sequences.copy()
    ## min_max trading
    price , trade,sentiments =np.split(sequences,[4,5],axis=2)
    trade , min_max_trading =self.min_max(trade)
    price , min_max_list =self.min_max(price)
    combine =np.concatenate([price,trade,sentiments],axis =2)
    scale_result = self.make_split(combine)
    original_result = self.make_split(original)
    ## min_max another
    result_list= []
    # for i in range(0,window_number):
    #   a, b, c= np.split(combine,[i,i+window_size+predict_size],axis=1)
    #   result_list.append(b)
    # result_array = np.array(result_list)
    # train ,vaild = np.split(result_array,[-1],axis=0)
    # # train_x,train_y=np.split(train.reshape(total_day%(window_size+predict_size)*ticker_number,window_size+predict_size,feature_size),[window_size],axis=1)
    # train_x,train_y=np.split(train.reshape((window_number-1)*ticker_number,window_size+predict_size,feature_size),[window_size],axis=1)
    # vaild_x,vaild_y=np.split(vaild.reshape(1*ticker_number,window_size+predict_size,feature_size),[window_size],axis=1)
    print("전처리 완료..")
    return original_result,scale_result ,min_max_list
