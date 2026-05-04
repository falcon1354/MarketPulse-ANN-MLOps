import yfinance as yf
import feedparser
import pandas as pd
from datetime import datetime

# -------- STOCK DATA --------
def fetch_stock_data():
    print("Fetching stock data...")
    stock = yf.download("AAPL", period="1d", interval="5m")

    if stock.empty:
        print("No stock data fetched!")
        return pd.DataFrame()

    stock.reset_index(inplace=True)
    stock = stock[['Datetime', 'Close']]
    stock.columns = ['timestamp', 'price']

    print("Stock data fetched:", stock.shape)
    return stock


# -------- NEWS DATA --------
def fetch_news():
    print("Fetching news data...")

    # Reliable RSS feed (BBC Business News)
    url = "http://feeds.bbci.co.uk/news/business/rss.xml"
    feed = feedparser.parse(url)

    news_data = []

    for entry in feed.entries[:20]:
        time = entry.get("published", None) or entry.get("updated", None)

        if time is None:
            continue

        news_data.append({
            "timestamp": time,
            "headline": entry.title
        })

    df = pd.DataFrame(news_data)

    if df.empty:
        print("⚠️ Still no news data fetched!")
    else:
        print("✅ News data fetched:", df.shape)

    return df

# -------- MERGE --------
def merge_data(stock_df, news_df):
    print("Merging data...")

    if stock_df.empty or news_df.empty:
        print("⚠️ One of the datasets is empty. Cannot merge.")
        return pd.DataFrame()

    # Convert timestamps safely
    stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'], errors='coerce', utc=True)
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], errors='coerce', utc=True)

    # ❗ REMOVE NULLS (THIS FIXES YOUR ERROR)
    stock_df = stock_df.dropna(subset=['timestamp'])
    news_df = news_df.dropna(subset=['timestamp'])

    # Sort (required for merge_asof)
    stock_df = stock_df.sort_values('timestamp')
    news_df = news_df.sort_values('timestamp')

    # Reset index (good practice)
    stock_df = stock_df.reset_index(drop=True)
    news_df = news_df.reset_index(drop=True)

    merged = pd.merge_asof(
        news_df,
        stock_df,
        on='timestamp',
        direction='nearest'
    )

    print("✅ Merged data shape:", merged.shape)
    return merged


# -------- MAIN --------
if __name__ == "__main__":
    stock_df = fetch_stock_data()
    news_df = fetch_news()

    print("Stock DF columns:", stock_df.columns)
    print("News DF columns:", news_df.columns)

    final_df = merge_data(stock_df, news_df)

    if not final_df.empty:
        final_df.to_csv("data/raw_data.csv", index=False)
        print("Data saved to data/raw_data.csv")
    else:
        print("No data saved due to errors")