import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
	def __init__(self, output_size, hidden_size, embedding_length,num_layers ,fully_layer_size,dropout):
		super(LSTM, self).__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.output_size = output_size
		self.hidden_size =hidden_size
		self.fully_layer_size = fully_layer_size
		self.embedding_length = embedding_length
		self.num_layers = num_layers
		self.lstm = nn.LSTM(embedding_length, hidden_size,num_layers =num_layers,batch_first=True,dropout=dropout) # Our main hero for this tutorial
		self.fc_1 =  nn.Linear(hidden_size,self.fully_layer_size ) # fully connected
		self.fc_2 = nn.Linear(self.fully_layer_size, 30) # fully connected last laye
		self.relu = nn.ReLU()
		self.hidden = self.reset_hidden_state()

	def reset_hidden_state(self):
		h_0 = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(self.device)
		c_0 = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(self.device)
		return (h_0,c_0)

	def forward(self, x):
		output, (hn, cn) = self.lstm(x, self.hidden) # (input, hidden, and internal state)
		# hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
		out = self.relu(output[:,-1])
		out = self.fc_1(out) # first dense
		out = self.relu(out) # relu
		out = self.fc_2(out) # final output
		return out