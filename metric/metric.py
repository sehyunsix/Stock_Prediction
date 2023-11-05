class Metric:
  def __init__(self, actual, predicted):
    self.actual = actual
    self.predicted = predicted

  def accuracy(self):
    correct = 0
    for i in range(len(self.actual)):
      if self.actual[i] == self.predicted[i]:
        correct += 1
    return correct / float(len(self.actual)) * 100.0

  def percentage(self, value):
    return value / float(len(self.actual)) * 100.0

  def risk(self):
    risk = 0
    for i in range(len(self.actual)):
      if self.actual[i] != self.predicted[i]:
        risk += 1
    return risk / float(len(self.actual)) * 100.0
