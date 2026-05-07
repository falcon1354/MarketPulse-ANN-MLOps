import yfinance as yf
import feedparser
import pandas as pd
import time

# Expand to multiple high-volume tickers
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

# -------- STOCK DATA --------
def fetch_stock_data(ticker):
    print(f"Fetching stock data for {ticker}...")
    stock = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)

    if stock.empty:
        return pd.DataFrame()

    stock = stock.reset_index()

    if "Datetime" in stock.columns:
        stock = stock[["Datetime", "Close"]]
        stock.columns = ["timestamp", "price"]
    elif "Date" in stock.columns:
        stock = stock[["Date", "Close"]]
        stock.columns = ["timestamp", "price"]
    
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = ["_".join(c).strip("_") for c in stock.columns]

    stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True)
    stock["ticker"] = ticker
    return stock.dropna()


# -------- NEWS DATA --------
def fetch_news(ticker):
    print(f"Fetching news data for {ticker}...")
    news_data = []

    # yfinance ticker news
    try:
        tk = yf.Ticker(ticker)
        yf_news = tk.news or []
        for item in yf_news:
            ts = item.get("providerPublishTime")
            title = item.get("title", "")
            if ts and title:
                news_data.append({
                    "timestamp": pd.Timestamp(ts, unit="s", tz="UTC"),
                    "headline": title,
                })
    except: pass

    # RSS feeds
    rss_feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "http://feeds.bbci.co.uk/news/business/rss.xml"
    ]

    for url in rss_feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:50]:
                time_str = entry.get("published") or entry.get("updated")
                title = getattr(entry, "title", "")
                if time_str and title:
                    ts = pd.to_datetime(time_str, utc=True, errors="coerce")
                    if pd.notna(ts):
                        news_data.append({"timestamp": ts, "headline": title})
        except: pass

    df = pd.DataFrame(news_data).drop_duplicates(subset=["headline"])
    return df


# -------- MAIN --------
if __name__ == "__main__":
    all_data = []
    
    for ticker in TICKERS:
        stock_df = fetch_stock_data(ticker)
        news_df = fetch_news(ticker)
        
        if not stock_df.empty and not news_df.empty:
            stock_df = stock_df.sort_values("timestamp")
            news_df = news_df.sort_values("timestamp")
            
            merged = pd.merge_asof(
                stock_df,
                news_df,
                on="timestamp",
                direction="nearest"
            )
            all_data.append(merged)
            print(f"Merged {ticker}: {len(merged)} rows")
        
        time.sleep(1) # Be nice to APIs

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv("data/raw_data.csv", index=False)
        print(f"Total dataset saved: {len(final_df)} rows across {len(TICKERS)} tickers")
    else:
        print("No data collected.")