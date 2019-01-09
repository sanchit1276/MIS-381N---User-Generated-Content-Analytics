#!/usr/bin/python3.6
import instaloader
import  time
import  pandas as pd
from datetime import datetime
from itertools import dropwhile, takewhile

L = instaloader.Instaloader()
df=pd.DataFrame()
posts = instaloader.Profile.from_username(L.context, 'natgeo').get_posts()
i=0
for post in posts:
    df = df.append({'Caption': post.caption, 'Likes': post.likes, 'URL': post.url  }, ignore_index=True)
    df.to_excel("Insta_withoutcomments.xlsx",index=False)
    i = i+1
    if i>400:
        break
print("Written to Insta_withoutcomments.xlsx file")