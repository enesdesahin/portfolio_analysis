import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from utils.data_loader import load_price_data, fetch_asset_metadata
from utils.returns import compute_returns
from utils.social import render_social_links



st.title("Builder")
st.markdown("""
<div style="margin-top: -10px; opacity: 0.6; margin-bottom: -20px;">
    Define your investment universe and portfolio weights
</div>
""", unsafe_allow_html=True)

st.divider()

# Initialize session state for available tickers
if "available_tickers" not in st.session_state:
    st.session_state.available_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "PG"]

if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = ["AAPL", "MSFT", "AMZN"]

def add_ticker():
    # Helper to add new tickers
    if "new_ticker_input" in st.session_state:
        new_ticker = st.session_state.new_ticker_input.strip().upper()
        if new_ticker:
            if new_ticker not in st.session_state.available_tickers:
                try:
                    tick = yf.Ticker(new_ticker)
                    hist = tick.history(period="1d")
                    if not hist.empty:
                        st.session_state.available_tickers.append(new_ticker)
                        st.session_state.selected_tickers.append(new_ticker)
                        st.toast(f"Added {new_ticker} to options")
                        st.session_state.new_ticker_input = ""
                    else:
                        st.toast(f"Could not find data for {new_ticker}")
                except Exception:
                    st.toast(f"Error validating {new_ticker}")
            else:
                 if new_ticker not in st.session_state.selected_tickers:
                     st.session_state.selected_tickers.append(new_ticker)
                     st.toast(f"{new_ticker} selected")
                 st.session_state.new_ticker_input = ""

# --- SIDEBAR INPUTS ---

tickers = st.sidebar.multiselect(
    "Select assets",
    options=st.session_state.available_tickers,
    default=st.session_state.selected_tickers,
    key="ticker_multiselect"
)
if tickers != st.session_state.selected_tickers:
    st.session_state.selected_tickers = tickers

st.sidebar.text_input(
    "Search & add ticker",
    key="new_ticker_input",
    on_change=add_ticker,
    placeholder="e.g. SPY, BTC-USD",
    help="Type a ticker and hit Enter to add it to the list."
)

if len(tickers) == 0:
    st.warning("Please select at least one asset")
    st.stop()
    
weighting_method = st.sidebar.selectbox("Weighting method", options=["Equal weights", "Manual weights"])

default_start = pd.to_datetime("2018-01-01")
default_end = pd.to_datetime("today")
date_range = st.sidebar.date_input("Date range", value=(default_start, default_end))

if len(date_range) != 2:
    st.warning("Please select both a start and end date.")
    st.stop()
start_date, end_date = date_range
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()


render_social_links()

# --- MAIN LOGIC (Builder View) ---

# Fetch Metadata early for display
metadata_df = fetch_asset_metadata(tickers)

weights_vector = []
valid_weights = False

if weighting_method == "Equal weights":
    n = len(tickers)
    w_val = 1.0 / n
    weights_vector = np.array([w_val] * n)
    
    st.markdown('<h3 style="margin-top: -20px;">Weights</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px;">
        Equal allocation across all selected assets
    </div>
    """, unsafe_allow_html=True)
    
    weights_df = pd.DataFrame({"Asset": tickers, "Weight": weights_vector})
    
    # Merge Metadata
    # Merge Metadata
    if not metadata_df.empty:
        weights_df = weights_df.merge(metadata_df, left_on="Asset", right_index=True, how="left")
        
    # Rename Asset -> Ticker
    weights_df.rename(columns={"Asset": "Ticker"}, inplace=True)
    
    # Desired Column Order: Company Name, Ticker, [Others], Weight
    desired_order = ["Company Name", "Ticker"]
    base_cols = [c for c in weights_df.columns if c not in desired_order and c != "Weight"]
    final_cols = desired_order + base_cols + ["Weight"]
    
    # Filter only existing columns just in case
    final_cols = [c for c in final_cols if c in weights_df.columns]
    weights_df = weights_df[final_cols]
        
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}), hide_index=True, use_container_width=True)
    valid_weights = True

else: # Manual weights
    st.markdown('<h3 style="margin-top: -20px;">Adjust Weights</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px;">
        Custom allocation. Ensure total sums to 100%.
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(len(tickers))
    manual_weights = []
    
    for i, ticker in enumerate(tickers):
        default_w = 1.0 / len(tickers)
        w = cols[i % len(cols)].slider(
            f"{ticker}",
            min_value=0.0,
            max_value=1.0,
            value=default_w,
            step=0.01
        )
        manual_weights.append(w)
    
    weights_vector = np.array(manual_weights)
    total_weight = weights_vector.sum()
    st.write(f"Total Weight: {total_weight:.2%}")
    
    # Display Table for Manual Weights too
    weights_df = pd.DataFrame({"Asset": tickers, "Weight": weights_vector})
    
    # Merge Metadata
    # Merge Metadata
    if not metadata_df.empty:
        weights_df = weights_df.merge(metadata_df, left_on="Asset", right_index=True, how="left")
    
    # Rename Asset -> Ticker
    weights_df.rename(columns={"Asset": "Ticker"}, inplace=True)
    
    # Desired Column Order: Company Name, Ticker, [Others], Weight
    desired_order = ["Company Name", "Ticker"]
    base_cols = [c for c in weights_df.columns if c not in desired_order and c != "Weight"]
    final_cols = desired_order + base_cols + ["Weight"]
    
    # Filter only existing columns
    final_cols = [c for c in final_cols if c in weights_df.columns]
    weights_df = weights_df[final_cols]
    
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}), hide_index=True, use_container_width=True)
    
    if abs(total_weight - 1.0) <= 0.01:
        valid_weights = True
        st.success("Weights are valid")
    else:
        st.error(f"Weights must sum to 100% (±1%). Current : {total_weight:.2%}")
        valid_weights = False

if st.button("Create portfolio", disabled=not valid_weights):
    with st.spinner("Creating Portfolio..."):
        prices = load_price_data(tickers, start_date, end_date)
        
        if prices.empty:
            st.error("No data found for selected assets.")
        else:
            asset_returns = compute_returns(prices)
            available_tickers = [t for t in tickers if t in asset_returns.columns]
            
            if len(available_tickers) != len(tickers):
                st.warning("Some assets failed to load data.")
                
            asset_returns = asset_returns[tickers] 
            portfolio_returns = asset_returns.dot(weights_vector)
            
            # Fetch Metadata (Sector, Country)
            metadata = fetch_asset_metadata(tickers)

            # Store Data
            st.session_state["portfolio"] = {
                "tickers": tickers,
                "weights": weights_vector,
                "prices": prices,
                "asset_returns": asset_returns,
                "portfolio_returns": portfolio_returns,
                "start_date": start_date,
                "end_date": end_date,
                "metadata": metadata,
                "show_toast": True
            }
            
            # Redirect to Analytics
            st.switch_page("pages/portfolio_analytics.py")
