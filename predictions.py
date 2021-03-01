from config import reddit_api
from keras.models import load_model
import pandas as pd
import preprocessing
from pickle import load
from praw import Reddit
import twint

model = load_model('final_model/model.h5')
model.load_weights('final_model/model_weights.h5')
vectorizer = load(open('final_model/vectorizer.pk', 'rb'))

def get_tweets(search, limit=1000): # get ~1000 most recent tweets from hashtag
    c = twint.Config()
    c.Limit = limit*2 # searching by language does not work, reducing to English-only reduces amount by about half
    c.Min_likes = 5
    c.Pandas = True
    c.Lang = 'en'
    c.Hide_output = True
    c.Search = search
    
    twint.run.Search(c)
    tweets = twint.storage.panda.Tweets_df
    tweets = tweets.loc[tweets['language']=='en']
    
    return tweets['tweet'].to_list()

def get_comments(url): # get all top-level coments from reddit thread
    reddit = Reddit(client_id=reddit_api['client_id'],
                    client_secret=reddit_api['client_secret'],
                    user_agent=reddit_api['user_agent'])

    submissionId = url[url.find('comments'):].split('/')[1]
    submission = reddit.submission(submissionId)
    submission.comments.replace_more(limit=None)
    comments = []
    for comment in submission.comments:
        comments.append(comment.body)
        
    return comments

def process_data(X): # run all preprocessing functions
    X_pre = list(map(preprocessing.remove_markdown, X))
    X_pre = list(map(preprocessing.remove_punctuation, X_pre))
    X_pre = list(map(preprocessing.tokenize, X_pre))
    X_pre = list(map(preprocessing.lemmatize, X_pre))
    X_join = [' '.join(x) for x in X_pre]
    
    return X_join

def get_predictions(source, limit=1000): # return pandas dataframe of review, prediction value, and predicted positive/negative sentiment
    if 'reddit.com' in source:
        data = get_comments(source)
    else:
        data = get_tweets(source, limit)
    X = process_data(data)
    
    X_processed = vectorizer.transform(process_data(X)).todense()
    y_pred = model.predict(X_processed)
    df = pd.DataFrame(zip(data, y_pred.flatten()), columns=['review', 'prediction'])
    df['positive'] = df['prediction'].map(lambda x: x>.6)
    
    return df

def get_web_output(source, limit=1000, samples=5):
    df = get_predictions(source, limit)
    value_counts = df.positive.value_counts(normalize=True)
    pos_percentage = round(value_counts[True]*100)
    neg_percentage = round(value_counts[False]*100)
    pos_samples = df[df['positive']].sample(5)['review'].tolist()
    neg_samples = df[~df['positive']].sample(5)['review'].tolist()
    
    return pos_percentage, neg_percentage, pos_samples, neg_samples