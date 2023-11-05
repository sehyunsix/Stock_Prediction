import os
import requests

class baseBot:
  def __init__(self, exchange):
    self.exchange = exchange
    self.models = {}
    self.load_models()

  def load_models(self):
    model_dir = f"/home/ssu36/tiger/Stock_Prediction/tradingbots/models/{self.exchange}"
    for filename in os.listdir(model_dir):
      if filename.endswith(".h5"):
        model_name = filename.split(".")[0]
        model_path = os.path.join(model_dir, filename)
        self.models[model_name] = load_model(model_path)

  def predict_prices(self, symbol):
    prices = {}
    for model_name, model in self.models.items():
      price = model.predict(symbol)
      prices[model_name] = price
    return prices

  def get_current_prices(self, symbol):
    url = f"https://{self.exchange}.com/api/v1/ticker?symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    return data["last_price"]

  def calculate_optimal_prices(self, symbol):
    predicted_prices = self.predict_prices(symbol)
    current_price = self.get_current_prices(symbol)
    buy_price = max(predicted_prices.values())
    sell_price = min(predicted_prices.values())
    if buy_price > current_price:
      buy_price = current_price
    if sell_price < current_price:
      sell_price = current_price
    return buy_price, sell_price

  def buy(self, symbol, quantity):
    buy_price, _ = self.calculate_optimal_prices(symbol)
    url = f"https://{self.exchange}.com/api/v1/buy?symbol={symbol}&price={buy_price}&quantity={quantity}"
    response = requests.post(url)
    return response.json()

  def sell(self, symbol, quantity):
    _, sell_price = self.calculate_optimal_prices(symbol)
    url = f"https://{self.exchange}.com/api/v1/sell?symbol={symbol}&price={sell_price}&quantity={quantity}"
    response = requests.post(url)
    return response.json()
