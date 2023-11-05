import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):
  def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
    super(TCN, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.num_channels = num_channels
    self.kernel_size = kernel_size
    self.dropout = dropout

    self.conv_layers = nn.ModuleList()
    self.batch_norm_layers = nn.ModuleList()
    self.relu_layers = nn.ModuleList()
    self.dropout_layers = nn.ModuleList()

    for i, channels in enumerate(num_channels):
      dilation = 2 ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      conv_layer = nn.Conv1d(
        in_channels=input_size if i == 0 else num_channels[i-1],
        out_channels=channels,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding
      )
      self.conv_layers.append(conv_layer)
      self.batch_norm_layers.append(nn.BatchNorm1d(channels))
      self.relu_layers.append(nn.ReLU())
      self.dropout_layers.append(nn.Dropout(dropout))

    self.fc_layer = nn.Linear(num_channels[-1], output_size)

  def forward(self, x):
    # x shape: (batch_size, input_size, sequence_length)
    for i in range(len(self.conv_layers)):
      x = self.conv_layers[i](x)
      x = self.batch_norm_layers[i](x)
      x = self.relu_layers[i](x)
      x = self.dropout_layers[i](x)

    # x shape: (batch_size, num_channels[-1], sequence_length)
    x = x.permute(0, 2, 1)
    # x shape: (batch_size, sequence_length, num_channels[-1])
    x = x.mean(dim=1)
    # x shape: (batch_size, num_channels[-1])
    x = self.fc_layer(x)
    # x shape: (batch_size, output_size)
    return x
