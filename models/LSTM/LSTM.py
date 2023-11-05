import yfinance as yf

def load_stock_data(stock_name, duration):
  # Define the start and end dates based on the duration
  if duration == '1d':
    start_date = end_date = 'today'
  elif duration == '1mo':
    start_date = '1 month ago'
    end_date = 'today'
  elif duration == '3mo':
    start_date = '3 months ago'
    end_date = 'today'
  elif duration == '6mo':
    start_date = '6 months ago'
    end_date = 'today'
  elif duration == '1y':
    start_date = '1 year ago'
    end_date = 'today'
  elif duration == '2y':
    start_date = '2 years ago'
    end_date = 'today'
  elif duration == '5y':
    start_date = '5 years ago'
    end_date = 'today'
  else:
    raise ValueError('Invalid duration specified')

  # Load the stock data using yfinance
  stock_data = yf.download(stock_name, start=start_date, end=end_date)

  # Extract the price data
  price_data = stock_data['Close']

  return price_data
