from dotenv import dotenv_values
from read_reddit import RedditAnalyzer
import streamlit as st


config = dotenv_values(".env")

# initialise the object
reddit_analyzer = RedditAnalyzer(config)

# collect the search text from user
topic = st.text_input('Search Topic', 'ipl')

# call the fetch comment method from class
comments = reddit_analyzer.fetch_comments(topic)
st.write("## Comments Collected")
st.write(comments)

st.write("## Sentiment of Each Comments")
comment_sentiments={}
for c in comments:
    sia_sentiment = reddit_analyzer.analyze_sentiment(c)
    textblob_sentiment = reddit_analyzer.analyze_sentiment_1(c)
    comment_sentiments[c] = {'SIA':sia_sentiment,'TextBlob':textblob_sentiment}
st.write(comment_sentiments)


# call the rpeprocessing function
preprocessed_comments = reddit_analyzer.preprocess_comments(comments)
st.write("## Comments after Preprocessing")
st.write(preprocessed_comments)

# identify viewpoints 
viewpoints = reddit_analyzer.identify_viewpoints(preprocessed_comments)
st.write("## Viewpoints for Comments")
st.write(viewpoints)


# call function to analyse sentiment and summarize viewpoints
summaries = reddit_analyzer.summarize_viewpoints(viewpoints)


st.write("## Summary of each viewpoint")
# Streamlit output
for viewpoint, summary_data in summaries.items():
    st.write(f"Viewpoint {viewpoint}:")
    st.write(f"Sentiment (SIA): {summary_data['sentiment_0']}")
    st.write(f"Sentiment (TextBlob): {summary_data['sentiment_1']}")
    st.write(f"Number of comments: {summary_data['comments']}")
    st.write(f"Summary: {summary_data['summary']}")
