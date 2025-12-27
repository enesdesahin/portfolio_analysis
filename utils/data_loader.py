import yfinance as yf
import streamlit as st
import pandas as pd

@st.cache_data
def load_price_data(tickers, start_date, end_date):
    """
    Fetches historical price data for the given tickers.
    Returns Adjusted Close prices only.
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        
        # Handle yfinance multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            # If we have multiple levels (Price, Ticker), we want 'Close' or 'Adj Close'
            # yf.download with auto_adjust=True usually returns 'Close' as the single accessible level if not flattened properly or multi-level.
            # Recent yfinance versions return MultiIndex (Price, Ticker).
            
            # Let's try to get 'Close' level
            try:
                return data['Close']
            except KeyError:
                # If 'Close' is not in the first level, maybe it's flattened? 
                # This depends heavily on yfinance version. 
                # Let's fallback to just returning data if it looks right, or specific check
                pass

        # Fallback simplistic check
        if 'Close' in data.columns:
             return data['Close']
             
        return data

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_asset_metadata(tickers):
    """
    Fetches metadata (Sector, Country) for the given tickers.
    """
    metadata = {}
    for t in tickers:
        try:
            # yfinance info fetching is synchronous and can be slow for many tickers
            # We use it carefully.
            info = yf.Ticker(t).info
            metadata[t] = {
                "Company Name": info.get("longName", t),
                "Sector": info.get("sector", "N/A"),
                "Country": info.get("country", "N/A")
            }
        except Exception:
            metadata[t] = {
                "Company Name": t, 
                "Asset Type": "N/A", 
                "Sector": "N/A", 
                "Country": "N/A"
            }
            
    return pd.DataFrame(metadata).T

@st.cache_data
def fetch_benchmark_data(start_date, end_date, benchmark_ticker="^GSPC"):
    """
    Fetches historical data for the benchmark (default: S&P 500).
    """
    return load_price_data([benchmark_ticker], start_date, end_date)
