import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
import networkx as nx

mpl.use('macosx')
duration = 60

df = pd.read_csv('2017-end2021.csv')
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
df = df.set_index('created_utc')

print('Dataframe length: {}'.format(len(df)))
print('Number of users: {}'.format(df['author'].nunique()))

# df = df.drop(df[(df.sentiment < 0.05) & (df.sentiment > -0.05)].index)  # Uncomment to remove zero elements (ie without sentiment)

df2 = df.resample("1d").mean().rolling(window=duration, min_periods=10, win_type='hanning').mean()
df2['sentiment_sem'] = df.resample("1d")['sentiment'].mean().rolling(window=duration, min_periods=10).sem()
df2['count'] = df.resample('1d').apply({'score': 'count'})

plt.plot(df2['sentiment'])
plt.fill_between(df2.index,
                 df2['sentiment'] - df2['sentiment_sem'],
                 df2['sentiment'] + df2['sentiment_sem'],
                 alpha=0.5, )
plt.ylabel('Sentiment', fontsize=18)
plt.yticks(np.arange(0, 0.5, 0.1))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.twinx()
plt.bar(df2.index, df2['count'], alpha=0.2, color='k')
plt.ylabel('Daily comment count', fontsize=18)

ax = plt.gca()
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlim([dt.date(2018, 3, 30), dt.date(2021, 12, 31)])
ax.tick_params(axis='both', which='major', labelsize=18)

fig = plt.gcf()
fig.set_size_inches(12, 8)
fig.savefig('test.pdf', dpi=300)
plt.show()

plt.boxplot(df['sentiment'])
plt.show()

# Correlation between sentiment & score (upvotes)
z = np.polyfit(df['sentiment'], df['score'], 1)
p = np.poly1d(z)
plt.scatter(df['sentiment'], df['score'])
plt.plot(df['sentiment'], p(df['sentiment']), 'r--')
plt.show()

print('')

# Number of comments per user
comment_counts = df['author'].value_counts()
plt.plot(comment_counts.values)
plt.xlabel('Rank')
plt.ylabel('Number of comments')
plt.show()


# Additonally can set to log-log plot with plt.xscale('log')


def user_rank(df, username):
    print('sentiment: {:.2f}'.format(df.loc[df['author'] == username]['sentiment'].mean()))
    a = df['author'].value_counts()
    print('number comments: {}'.format(a[username]))
    print('rank: {}'.format(np.where(a.values == a[username])[0][0] + 1))
