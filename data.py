import pandas as pd
df = pd.read_csv('dataset.csv')
df_filtered = df.dropna()
df_filtered.to_csv('dataset.csv', index=False)