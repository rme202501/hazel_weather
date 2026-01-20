import pandas as pd

data = pd.read_csv("merged_output.csv")

# Ensure the timestamp column is datetime, sorted, and monotonic for rolling windows
data["UTC_timestamp"] = pd.to_datetime(data["UTC_timestamp"])
data = data.sort_values("UTC_timestamp").reset_index(drop=True)

data["co2_rolling"] = data.rolling(window="1h", on="UTC_timestamp")["co2"].mean()
data["0.3um_rolling"] = data.rolling(window="1h", on="UTC_timestamp")["0.3um"].mean()

data.to_csv("nohistory_rolling_average.csv", index=False)

