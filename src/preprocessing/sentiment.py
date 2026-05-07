import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']

    if score >= 0.05:
        return "positive", score
    elif score <= -0.05:
        return "negative", score
    else:
        return "neutral", score


def process_data(input_path="data/raw_data.csv",
                 output_path="data/processed_data.csv"):

    df = pd.read_csv(input_path)

    sentiments = []
    scores = []

    for text in df["headline"]:
        label, score = get_sentiment(text)
        sentiments.append(label)
        scores.append(score)

    df["sentiment"] = sentiments
    df["sentiment_score"] = scores

    # -------- MARKET LABEL (IMPORTANT FOR ANN) --------
    df["price"] = pd.to_numeric(df["price"], errors='coerce')

    df["future_price"] = df["price"].shift(-1)

    # Calculate percentage change
    df["price_change_pct"] = (df["future_price"] - df["price"]) / df["price"]

    # Define meaningful movement threshold
    threshold = 0.001   # 0.1%

    df["label"] = df["price_change_pct"].apply(
        lambda x: 1 if x > threshold else 0
    )
    
    df = df.dropna()

    df.to_csv(output_path, index=False)

    print("Sentiment + labels created!")
    print("Saved to:", output_path)


if __name__ == "__main__":
    process_data()