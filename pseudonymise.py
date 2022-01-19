from faker import Faker
import numpy as np
import pandas as pd

# example dataframe
df = pd.read_csv('2017-end2021.csv')

df.head()
#      name
# 0     foo
# 1     bar
# 2     foo
# 3  foobar
# 4  foobar
# 5     NaN
# 6

# pseudonymization of the name column using Faker
fake = Faker()

# empty string values as nan
df.replace('', np.nan, inplace=True)

# name replacement dictionary
name_replacements = {name: fake.name().lower().replace(" ", "") for name in df['author'].unique()}

# apply replacement
df.replace({"author": name_replacements}, inplace=True)

df.head()

df.to_csv('data_anonymised.csv')
