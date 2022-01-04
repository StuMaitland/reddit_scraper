import praw
from psaw import PushshiftAPI
import datetime as dt
import pandas as pd
from helpers import *

r = praw.Reddit(
    client_id="-s1mxL-NBtZdx96cMhpVDw",
    client_secret="xTZxgGhZ-qffpT7KG3yKxO61gUZ2ZA",
    user_agent="testscript",
)
#
# for submission in reddit.subreddit("juniordoctorsuk").hot(limit=10):
#     print(submission.title)


api = PushshiftAPI(r)

start_epoch = int(dt.datetime(2021, 10, 29).timestamp())
end_epoch = int(dt.datetime(2021, 12, 31).timestamp())

gen = api.search_comments(after=start_epoch,
                          before=end_epoch,
                          subreddit='juniordoctorsuk',
                          )

max_response_cache = 10000
cache = []

df = pd.DataFrame()

for c in gen:
    safe_text = clean_tweet(c.body)
    sentiment = analize_sentiment(safe_text)
    df_row = pd.DataFrame([c.id, c.body, c.author, c.created_utc, c.score, c.parent_id, sentiment])
    df_row = df_row.transpose()
    df = df.append(df_row, ignore_index=True)

df.columns = ['id', 'body', 'author', 'created_utc', 'score', 'parent_id', 'sentiment']
df.to_csv('end_2021.csv', index=False)

print('')
