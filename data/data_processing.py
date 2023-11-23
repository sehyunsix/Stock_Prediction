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