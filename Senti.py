# Updated Twitter Sentiment Analysis Code

# %% Import required libraries
import tweepy
import re
from collections import Counter
import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline
import openai

# API keys and tokens
API_KEY = "zstDwwo8OGirrWPUsFGU6X5Ku"
API_SECRET = "8IASLdHNq3LvhUilFjrhOrhOp7aQsKfsLxVcgydIBpOfCQiddR"
ACCESS_TOKEN = "1838865367541985280-VpMHFfcD8RPNvI199zwIjdPUWZ5DBa"
ACCESS_TOKEN_SECRET = "hwZnQdD6CqHOvm9MPRCZGX8MRdZrrp2RyLPEaiiQCCHN7"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMkfygEAAAAAlXQYqa0Nvy6hozsHGEUiTXXR1uI%3DKrXLUW34Xynrb9v5Tymy0ZQesYRz9oBCHVD6U5LKMqNq6xybGE"


OPENAI_API_KEY = "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"  # Replace with your OpenAI API key

# Authenticate to Twitter
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Set up the OpenAI API key (Make sure to keep your API key secure)
openai.api_key = OPENAI_API_KEY

# Function to generate Jeremy Clarkson's review using OpenAI's chat API
def clarkson_review_with_chatgpt(sentiment, company):
    prompt = f"Imagine Jeremy Clarkson is giving a review of the {company} based on the public sentiment. The sentiment is {sentiment}. Write a review in the style of Jeremy Clarkson, known for his sarcastic, humorous, and often exaggerated tone. Be critical and witty while analyzing the car and its features, but also show some praise if the sentiment is positive."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can also use 'gpt-3.5-turbo' if you prefer
            messages=[
                {"role": "system", "content": "You are Jeremy Clarkson."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # Limit the length of the response
            temperature=0.7  # Control the randomness of the response
        )

        return response['choices'][0]['message']['content'].strip()
    
    except Exception as e:
        return f"Error generating review: {e}"

# Function to talk about how the car handles on different terrains
def terrain_handling_review(company):
    prompt = f"Imagine Jeremy Clarkson is reviewing how the {company} performs on different terrains like wet roads, snowy roads, hilly roads, and desert roads. Write the review in Jeremy Clarkson's sarcastic, witty, and exaggerated style, with a humorous take on how the car handles each type of terrain."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can also use 'gpt-3.5-turbo' if you prefer
            messages=[
                {"role": "system", "content": "You are Jeremy Clarkson."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  # Limit the length of the response
            temperature=0.7  # Control the randomness of the response
        )

        return response['choices'][0]['message']['content'].strip()
    
    except Exception as e:
        return f"Error generating terrain handling review: {e}"

# Streamlit UI
st.title("Twitter Sentiment Analysis and Trend Prediction")
company = st.selectbox("Select Company", ["Tesla", "Ferrari", "Toyota", "Honda"])
query = company

if st.button("Fetch Tweets"):
    try:
        # Fetch tweets
        response = client.search_recent_tweets(query=query, max_results=50, tweet_fields=["public_metrics", "text"])
        tweets = response.data if response.data else []

        if not tweets:
            st.warning("No tweets found!")
        else:
            st.success("Tweets fetched successfully!")

            # Process tweets
            tweet_texts = [tweet.text for tweet in tweets]
            cleaned_tweets = [re.sub(r"http\S+|@\w+|[^A-Za-z0-9\s]", "", text).strip() for text in tweet_texts]

            # Sentiment Analysis
            sentiment_pipeline = pipeline("sentiment-analysis")
            sentiments = sentiment_pipeline(cleaned_tweets)
            sentiment_labels = [result["label"] for result in sentiments]

            # Display Sentiment Results
            st.subheader("Sentiment Distribution")
            counts = Counter(sentiment_labels)
            fig, ax = plt.subplots()
            ax.bar(counts.keys(), counts.values(), color=['green', 'red', 'gray'])
            st.pyplot(fig)

            # Emoji and Hashtag Analysis
            emojis = Counter([char for text in tweet_texts for char in text if char in re.findall(r"[^\w\s,]", text)])
            hashtags = Counter([word for text in tweet_texts for word in text.split() if word.startswith("#")])

            st.subheader("Emoji Sentiment Analysis")
            if emojis:
                emoji_fig, emoji_ax = plt.subplots()
                emoji_ax.bar(emojis.keys(), emojis.values(), color='blue')
                st.pyplot(emoji_fig)
            else:
                st.write("No emojis found in tweets.")

            st.subheader("Hashtag Frequency")
            if hashtags:
                hashtag_fig, hashtag_ax = plt.subplots()
                hashtag_ax.bar(hashtags.keys(), hashtags.values(), color='purple')
                st.pyplot(hashtag_fig)
            else:
                st.write("No hashtags found in tweets.")

            # Tweet Impact Score
            impact_scores = []
            for tweet in tweets:
                metrics = tweet.public_metrics
                score = metrics["retweet_count"] * 2 + metrics["like_count"] * 1.5 + metrics["reply_count"]
                impact_scores.append((tweet.text, score))
            sorted_scores = sorted(impact_scores, key=lambda x: x[1], reverse=True)

            st.subheader("Top 5 Impactful Tweets")
            for text, score in sorted_scores[:5]:
                st.write(f"Impact Score: {score}")
                st.write(f"Tweet: {text}")
                st.write("---")

            # Sentiment-Driven Recommendations
            st.subheader("Recommendations")
            if counts["NEGATIVE"] > counts["POSITIVE"]:
                st.write(f"Public sentiment around {query} is more negative. Consider addressing common concerns like:")
                st.write("- Enhancing customer service.")
                st.write("- Improving product quality.")
                st.write("- Clarifying recent controversies.")
            elif counts["POSITIVE"] > counts["NEGATIVE"]:
                st.write(
                    f"Public sentiment around {query} is mostly positive. Keep building on these positive aspects:")
                st.write("- Highlighting successful milestones.")
                st.write("- Amplifying customer satisfaction stories.")
            else:
                st.write(f"Public sentiment around {query} is neutral. It may help to:")
                st.write("- Increase engagement through positive PR campaigns.")
                st.write("- Address any potential concerns proactively.")

            # Jeremy Clarkson's Review based on Sentiment
            st.subheader("Jeremy Clarkson's Review")

            # Generate Clarkson's review using ChatGPT
            if counts["NEGATIVE"] > counts["POSITIVE"]:
                clarkson_opinion = clarkson_review_with_chatgpt("negative", company)
            elif counts["POSITIVE"] > counts["NEGATIVE"]:
                clarkson_opinion = clarkson_review_with_chatgpt("positive", company)
            else:
                clarkson_opinion = clarkson_review_with_chatgpt("neutral", company)

            st.write(clarkson_opinion)

            # Jeremy Clarkson's Terrain Handling Review
            st.subheader(f"How the {company} Handles on Wet, Snowy, Hilly, and Desert Roads")

            # Generate terrain handling review using ChatGPT
            terrain_review = terrain_handling_review(company)

            st.write(terrain_review)

    except Exception as e:
        st.error(f"Try after 15 mins, Error fetching tweets: {e}")
        tweets = []