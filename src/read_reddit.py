import praw
import nltk
from nltk.cluster.util import cosine_distance
from nltk.cluster import KMeansClusterer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from transformers import pipeline
from dotenv import dotenv_values
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
# from summarizer import TransformerSummarizer


class RedditAnalyzer:
    def __init__(self, config):
        self.reddit = praw.Reddit(
            client_id=config['REDDIT_CLIENT_ID'],
            client_secret=config['REDDIT_CLIENT_SECRET'],
            password=config['REDDIT_PW'],
            user_agent=config['REDDIT_USER_AGENT'],
            username=config['REDDIT_UNAME']
        )
        # self.summarizer = pipeline("summarization")
        self.sia = SentimentIntensityAnalyzer()

    def fetch_comments(self, topic, limit=10):
        subreddit = self.reddit.subreddit(topic)
        comments = []
        for submission in subreddit.new(limit=limit):
            submission.comments.replace_more(limit=0)
            comments.extend([comment.body for comment in submission.comments.list()])
        return comments

    def preprocess_comments(self, comments):
        stop_words = stopwords.words('english')
        preprocessed_comments = [
            ' '.join([word for word in comment.lower().translate(str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")).split()
                      if word not in stop_words])
            for comment in comments
        ]
        return preprocessed_comments

    def identify_viewpoints(self, comments):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(comments)
        cluster_model = KMeansClusterer(num_means=3, distance=cosine_distance, repeats=5, avoid_empty_clusters=True)
        clusters = cluster_model.cluster(X.toarray(), assign_clusters=True)
        viewpoints = {}
        for cluster, comment in zip(clusters, comments):
            if cluster not in viewpoints:
                viewpoints[cluster] = []
            viewpoints[cluster].append(comment)
        return viewpoints

    def analyze_sentiment(self, text):
        # sia = SentimentIntensityAnalyzer()
        score = self.sia.polarity_scores(text)
        compound = score['compound']
        if compound > 0.05:
            return 'Positive'
        elif compound < -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_sentiment_1(self,text):
        blob = TextBlob(text)
        if blob.sentiment.polarity > 0:
            return 'Positive'
        elif blob.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def summarize_viewpoints(self, viewpoints):
        # summarizer = pipeline("summarization")
        # summarizer = TransformerSummarizer()
        summarizer = pipeline("summarization")
        summaries = {}
        for viewpoint, comments in viewpoints.items():
            text = ' '.join(comments)
            summary = summarizer(text, max_length=50, min_length=10)[0]["summary_text"]
            sentiment = self.analyze_sentiment(text)
            sentiment_1 = self.analyze_sentiment_1(text)
            summaries[viewpoint] = {
                'summary': summary,
                'sentiment_0': sentiment,
                'sentiment_1': sentiment_1,
                'comments': len(comments)
            }
        return summaries
    