import torch
import torch.nn as nn
import torch.optim as optim

class BaseTrainer(nn.Module):
  def __init__(self):
    super(BaseTrainer, self).__init__()

  def train(self, dataloader):
    self.model.train()
    train_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
      self.optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = self.loss_fn(outputs, targets)
      loss.backward()
      self.optimizer.step()
      train_loss += loss.item()
    return train_loss / len(dataloader)
