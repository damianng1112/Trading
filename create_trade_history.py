import pandas as pd

# Define the columns for your trade history
columns = ['timestamp', 'signal', 'price', 'size', 'stop_loss', 'take_profit', 'status', 
           'close_timestamp', 'close_price', 'close_reason', 'profit_loss_pct']

# Create empty DataFrame with these columns
empty_df = pd.DataFrame(columns=columns)

# Save to CSV
empty_df.to_csv("data/trade_history.csv", index=False)