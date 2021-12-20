import pandas as pd

df1 = pd.read_csv('test4.csv')
df2 = pd.read_csv('test5.csv')

df3 = pd.concat([df1,df2])

df3.to_csv('test6.csv')