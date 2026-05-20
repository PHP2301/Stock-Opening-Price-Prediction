import pandas as pd

print('=== VNM_processed.csv (lich su cu) ===')
df1 = pd.read_csv('data/VNM_processed.csv')
mask2015 = df1['date'].str.startswith('2015')
mask2023 = df1['date'].str.startswith('2023')
print(f'  2015 close: {df1[mask2015]["close"].min():.2f} ~ {df1[mask2015]["close"].max():.2f}')
print(f'  2023 close: {df1[mask2023]["close"].min():.2f} ~ {df1[mask2023]["close"].max():.2f}')
print(f'  First row:  {df1.iloc[0][["date","close","open"]].to_dict()}')

print()
print('=== VNM_prices.csv ===')
df2 = pd.read_csv('data/VNM_prices.csv')
print(f'  Columns: {df2.columns.tolist()}')
print(f'  First close (2019): {df2.iloc[0]["close"]:.2f}')
mask23 = df2['time'].str.startswith('2023')
if mask23.any():
    print(f'  2023 close avg:   {df2[mask23]["close"].mean():.2f}')
print(f'  Latest close:     {df2.iloc[-1]["close"]:.2f}  ({df2.iloc[-1]["time"]})')

print()
print('=== VNM.VN_processed.csv (Yahoo Finance hien tai) ===')
df3 = pd.read_csv('data/VNM.VN_processed.csv')
mask3 = df3['date'].str.startswith('2023')
print(f'  2023 close: {df3[mask3]["close"].min():.2f} ~ {df3[mask3]["close"].max():.2f}')
print(f'  Latest close: {df3.iloc[-1]["close"]:.2f}  ({df3.iloc[-1]["date"]})')
