import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
	def __init__(self, output_size, hidden_size, batch_size,input_size,num_layers ,fully_layer_size,dropout):
		super(LSTMModel, self).__init__()
		self.output_size = output_size
		self.hidden_size =hidden_size
		self.batch_size = batch_size
		self.fully_layer_size = fully_layer_size
		self.input_size = input_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size,num_layers =num_layers,batch_first=True,dropout=dropout) # Our main hero for this tutorial
		self.fc_1 =  nn.Linear(hidden_size,self.fully_layer_size ) # fully connected
		self.fc_2 = nn.Linear(self.fully_layer_size, output_size) # fully connected last laye
		self.relu = nn.ReLU()
		self.hidden = self.reset_hidden_state()

	def reset_hidden_state(self):
		h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
		c_0 = torch.zeros(self.num_layers, self.batch_size ,self.hidden_size)
		return (h_0,c_0)

	def forward(self, x):
		output, (hn, cn) = self.lstm(x, self.hidden) # (input, hidden, and internal state)
		# hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
		out = self.relu(output[:,-1])
		out = self.fc_1(out) # first dense
		out = self.relu(out) # relu
		out = self.fc_2(out) # final output
		return out