from typing import List, Dict
from dataclasses import dataclass
import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from .base import BaseTrainer, BaseTrainingArguments, collate_dictlist
from itertools import chain
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from pprint import pprint
from datasets import load_dataset
from models.TimeSeries.TCN.TCN import TCNModel
from sklearn.metrics import mean_absolute_error
from data.data_processing import Window_maker

@dataclass
class TCNModelArguments(BaseTrainingArguments):
      input_size : int = 5
      output_size : int = 15
      channels_num : int = 80
      channels_level: int =3
      kernel_size : int =7
      dropout : float = 0.25

class TCNTrainer(BaseTrainer):

  def get_model(self, args):
        kwargs = {}
        kwargs['input_size'] = args.input_size
        kwargs['output_size'] = args.output_size
        kwargs['num_channels'] = [args.channels_num]*args.channels_level
        kwargs['kernel_size'] = args.kernel_size
        kwargs['dropout'] = args.dropout
        model = TCNModel(**kwargs)
        return model


  def make_dataset(self,x_data,y_data,y_original,target,feature,min_max_list,train = True):
      class BaseDataset(Dataset):
        def __init__(self, x_data, y_data):
            x_data = np.swapaxes(x_data, 1, 2)
            y_data = np.swapaxes(y_data, 1, 2)
            label_data = np.swapaxes(y_original, 1, 2)
            self.train = train
            self.lable_data =torch.tensor(label_data[:,feature.index(target),:],dtype = torch.float32)
            self.x_data = torch.tensor(x_data,dtype =torch.float32)
            self.y_data = torch.tensor(y_data[:,feature.index(target),:],dtype = torch.float32)
            self.min_max_list = torch.tensor(min_max_list,dtype = torch.float32)
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index] ,self.lable_data[index] ,self.min_max_list[self.get_minmax_index(index)]
        def get_minmax_index(self,index):
            if self.train == True:
              index = index//650
            return index
        def __len__(self):
            return self.x_data.shape[0]
      return BaseDataset(x_data,y_data)


  def _shared_step(self, batch):
      x_train, y_train ,true_label, min_max= batch
      outputs = self.model(x_train)
      criterian = nn.MSELoss()
      predict = Window_maker.inverse_min_max(outputs,min_max)
      label =  Window_maker.inverse_min_max(y_train,min_max)
      mae = torch.mean(torch.abs(label-predict))
      acc = (1-torch.mean((torch.abs(label-predict)/label)))
      scale_check =torch.mean(torch.abs(true_label-label))
      loss =  criterian(label, predict)
      return {
          "loss": loss,
          "mae": mae,
          "acc": acc,
          "check": scale_check
      }


  def training_step(self, batch):
    loss = self._shared_step(batch)["loss"]
    return loss.mean()

  def evaluation_step(self, batch):
    return self._shared_step(batch)


  def pad_tensor_list(self, tensor_list):
      max_length = max(tensor.size(0) for tensor in tensor_list)
      if tensor_list[0].dim() == 1:
          return [F.pad(t, (max_length - t.size(0),0),'constant',0) if t.size(0) < max_length else t for t in tensor_list]
      else:
          return [F.pad(t, (0,0, max_length - t.size(0), 0),'constant',0) if t.size(0) < max_length else t for t in tensor_list]

  def collate_evaluation(self, results: List[Dict]):
        losses = torch.stack(results["loss"])
        accuracies = torch.stack(results["acc"])
        maes = torch.stack(results["mae"])
        checks = torch.stack(results["check"])
        torch.mean(losses).item()
        eval_results = {}
        eval_results["loss"] = losses.mean().item()
        eval_results["acc"] = accuracies.mean().item()
        eval_results["mae"] = maes.mean().item()
        eval_results["check"] = checks.mean().item()
        return eval_results

if __name__ == "__main__":
    TCNTrainer.main(TCNModelArguments)