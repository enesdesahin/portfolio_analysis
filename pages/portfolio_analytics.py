import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from utils.data_loader import load_price_data, fetch_asset_metadata
from utils.returns import compute_returns
from utils.risk import calculate_beta, calculate_alpha, calculate_tracking_error, calculate_information_ratio
from utils.factors import get_fama_french_factors
from utils.social import render_social_links



st.title("Analytics")
st.markdown("""
<div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px;">
    Analysis of your constructed portfolio
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR INPUTS ---

# 1. Benchmark Selection (Available globally for simplicity with Tabs)
# 1. Benchmark Selection (Available globally for simplicity with Tabs)
benchmark_ticker = st.sidebar.selectbox(
    "Select benchmark",
    options=["^GSPC", "^IXIC", "^RUT", "ACWI", "AGG", "SPY", "QQQ"],
    index=0
)



# 2. Date Range Sidebar (Reactive)
pf_data = st.session_state.get("portfolio")

if not pf_data:
    st.warning("Please build a portfolio first", icon=":material/error:")
    if st.button("Go to portfolio builder"):
        st.switch_page("pages/portfolio_builder.py")
    st.stop()

# Helper to get tickers for reactive update
tickers = pf_data["tickers"]

default_start = pd.to_datetime("2018-01-01")
default_end = pd.to_datetime("today")

current_start = pf_data.get("start_date", default_start)
current_end = pf_data.get("end_date", default_end)

date_range = st.sidebar.date_input("Date range", value=(current_start, current_end))

if len(date_range) != 2:
    st.warning("Select start and end date.")
    st.stop()
start_date, end_date = date_range

render_social_links()

# Reactive Date Update Logic
p_start = pf_data["start_date"]
p_end = pf_data["end_date"]

if (start_date != p_start or end_date != p_end):
     with st.spinner("Updating dates..."):
        # We re-fetch prices for the EXISTING tickers
        prices = load_price_data(tickers, start_date, end_date)
        if not prices.empty:
            asset_returns = compute_returns(prices)
            # Ensure order matches weights
            w_tickers = pf_data["tickers"]
            
            # Robust alignment
            available_cols = [t for t in w_tickers if t in asset_returns.columns]
            if len(available_cols) == len(w_tickers):
                 asset_returns = asset_returns[w_tickers]
                 portfolio_returns = asset_returns.dot(pf_data["weights"])
                 
                 # Update Session State
                 st.session_state["portfolio"].update({
                    "prices": prices,
                    "asset_returns": asset_returns,
                    "portfolio_returns": portfolio_returns,
                    "start_date": start_date,
                    "end_date": end_date
                 })
                 st.rerun()
            else:
                st.error("Data missing for some assets in new date range.")

# UNINDENTED BLOCK STARTS HERE
portfolio_returns = pf_data["portfolio_returns"]
curr_tickers = pf_data["tickers"]
curr_weights = pf_data["weights"]

if pf_data.get("show_toast", False):
    st.toast("Portfolio successfully created !")
    st.session_state["portfolio"]["show_toast"] = False

# --- MAIN TABS ---
main_tabs = st.tabs(["Portfolio analysis", "Benchmark comparison", "Factor models"])

# ==========================================
# TAB 1: PORTFOLIO ANALYSIS
# ==========================================
with main_tabs[0]:
    # --- KPI CALCULATIONS ---
    # 1. Cumulative Return
    cum_ret_series = (1 + portfolio_returns).cumprod()
    total_cum_return = cum_ret_series.iloc[-1] - 1
    
    # 2. Annualized Return (assuming daily returns)
    annualized_return = portfolio_returns.mean() * 252
    
    # 3. Annualized Volatility
    annualized_vol = portfolio_returns.std() * np.sqrt(252)
    
    # 4. Sharpe Ratio (assuming Rf=0 for simplicity in Tier 1)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    
    # 5. Maximum Drawdown
    drawdown = cum_ret_series / cum_ret_series.cummax() - 1
    max_drawdown = drawdown.min()
    
    # --- DISPLAY KPIS ---
    st.markdown('<h3 style="margin-top: -5px;">Key performance indicators</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Core risk and return metrics for the selected period
    </div>
    """, unsafe_allow_html=True)
    kpi_cols = st.columns(5)
    
    # Helper for custom metric card
    def kpi_card(title, subtitle, value):
        st.markdown(f"""
        <div style="
            border: 1px solid #7c7c7c;
            padding: 15px;
            border-radius: 0px;
            margin-bottom: 10px;
        ">
            <div style="font-size: 16px; font-weight: 600; margin-bottom: 2px;">{title}</div>
            <div style="font-size: 12px; opacity: 0.6; margin-bottom: 8px;">{subtitle}</div>
            <div style="font-size: 28px; font-weight: 500;">{value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_cols[0]:
        kpi_card("Cumulative return", "Total period growth", f"{total_cum_return:.2%}")
    with kpi_cols[1]:
        kpi_card("Annualized return", "Yearly average return", f"{annualized_return:.2%}")
    with kpi_cols[2]:
        kpi_card("Annualized volatility", "Annualized risk", f"{annualized_vol:.2%}")
    with kpi_cols[3]:
        kpi_card("Sharpe ratio", "Risk-adjusted return", f"{sharpe_ratio:.2f}")
    with kpi_cols[4]:
        kpi_card("Max drawdown", "Max loss from peak", f"{max_drawdown:.2%}")
    
    st.divider()
    
    # Display Weight Table & Pie Chart (Side-by-Side)
    st.markdown('<h3 style="margin-top: -20px;">Portfolio composition</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Breakdown of assets and their allocated weights
    </div>
    """, unsafe_allow_html=True)
    
    w_df = pd.DataFrame({
        "Asset": curr_tickers,
        "Weight": curr_weights
    })
    
    # Merge Metadata if available
    sector_weights = None
    country_weights = None
    
    if "metadata" in pf_data:
        meta_df = pf_data["metadata"]
        w_df = w_df.merge(meta_df, left_on="Asset", right_index=True, how="left")
        
        # Prepare aggregations
        if "Sector" in w_df.columns:
            sector_weights = w_df.groupby("Sector")["Weight"].sum().reset_index()
        if "Country" in w_df.columns:
            country_weights = w_df.groupby("Country")["Weight"].sum().reset_index()
    
    # Rename Asset -> Ticker
    w_df.rename(columns={"Asset": "Ticker"}, inplace=True)
    
    # Desired Column Order: Company Name, Ticker, [Others], Weight
    desired_order = ["Company Name", "Ticker"]
    base_cols = [c for c in w_df.columns if c not in desired_order and c != "Weight"]
    final_cols = desired_order + base_cols + ["Weight"]
    
    # Filter only existing columns
    final_cols = [c for c in final_cols if c in w_df.columns]
    w_df = w_df[final_cols]
    
    # Display Table (Full Width)
    st.dataframe(w_df.style.format({
        "Weight": "{:.2%}"
    }), hide_index=True, use_container_width=True)
    
    # Brand Palette (Green/Monochrome focus)
    brand_palette = ["#1ed760", "#5ce196", "#ffffff", "#b3b3b3", "#178c3e", "#0e5a26"]
    
    # Display 3 Pie Charts
    pie_cols = st.columns(3)
    
    # 1. Asset Weights
    with pie_cols[0]:
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom: 5px;">
                <div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">Asset allocation</div>
                <div style="font-size: 13px; opacity: 0.6;">Distribution by individual selection</div>
            </div>
            """, unsafe_allow_html=True)
            
            fig_asset = px.pie(w_df, values='Weight', names='Ticker', hole=0.4, color_discrete_sequence=brand_palette)
            fig_asset.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), 
                height=280, 
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_asset, use_container_width=True)
        
    # 2. Sector Weights
    with pie_cols[1]:
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom: 5px;">
                <div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">Sector allocation</div>
                <div style="font-size: 13px; opacity: 0.6;">Exposure across industry sectors</div>
            </div>
            """, unsafe_allow_html=True)
            
            if sector_weights is not None and not sector_weights.empty:
                fig_sec = px.pie(sector_weights, values='Weight', names='Sector', hole=0.4, color_discrete_sequence=brand_palette)
                fig_sec.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10), 
                    height=280, 
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_sec, use_container_width=True)
            else:
                st.info("No sector data.")
    
    # 3. Country Weights
    with pie_cols[2]:
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom: 5px;">
                <div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">Country allocation</div>
                <div style="font-size: 13px; opacity: 0.6;">Geographic distribution of assets</div>
            </div>
            """, unsafe_allow_html=True)
            
            if country_weights is not None and not country_weights.empty:
                fig_country = px.pie(country_weights, values='Weight', names='Country', hole=0.4, color_discrete_sequence=brand_palette)
                fig_country.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10), 
                    height=280, 
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_country, use_container_width=True)
            else:
                st.info("No country data.")
    
    # Display Charts (Tabs)
    # Display Charts (Tabs)
    tabs = st.tabs(["Performance", "Drawdown", "Volatility", "Correlation"])
    
    with tabs[0]:
        st.markdown("#### Performance")
        st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
            Growth of $1 invested over the specific period
        </div>
        """, unsafe_allow_html=True)
        
        # Convert Series to DataFrame for cleaner Plotly handling
        perf_df = pd.DataFrame(cum_ret_series)
        perf_df.columns = ["Cumulative return"]
        
        # Convert Series to DataFrame for cleaner Plotly handling
        perf_df = pd.DataFrame(cum_ret_series)
        perf_df.columns = ["Cumulative return"]
        
        fig_perf = px.area(perf_df, x=perf_df.index, y="Cumulative return")
        
        # Color based on final performance (Red if loss, Green if gain)
        final_val = cum_ret_series.iloc[-1]
        if final_val < 1.0:
            color = '#ff4b4b'
            fill = 'rgba(255, 75, 75, 0.2)'
        else:
            color = '#1ed760'
            fill = 'rgba(30, 215, 96, 0.2)'
            
        fig_perf.update_traces(line_color=color, fillcolor=fill)
        fig_perf.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Date",
            yaxis_title="Cumulative return",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        
    with tabs[1]:
        st.markdown("#### Drawdown")
        st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
            Decline from the historical peak value
        </div>
        """, unsafe_allow_html=True)
        
        # Convert Series to DataFrame
        dd_df = pd.DataFrame(drawdown)
        dd_df.columns = ["Drawdown"]
        
        fig_dd = px.area(dd_df, x=dd_df.index, y="Drawdown")
        fig_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
        fig_dd.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Date",
            yaxis_title="Drawdown",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    
    with tabs[2]:
        st.markdown("#### Rolling volatility")
        st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
            21-day annualized rolling volatility
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate Rolling Volatility (21 days)
        rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
        
        # Convert Series to DataFrame
        vol_df = pd.DataFrame(rolling_vol)
        vol_df.columns = ["Volatility"]
        
        fig_vol = px.area(vol_df, x=vol_df.index, y="Volatility")
        fig_vol.update_traces(line_color='#1ed760', fillcolor='rgba(30, 215, 96, 0.2)')
        fig_vol.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Date",
            yaxis_title="Annualized volatility",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with tabs[3]:
        st.markdown("#### Correlation matrix")
        st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
             Correlation between asset returns (1 = perfect positive correlation, -1 = perfect negative)
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate Correlation
        if "asset_returns" in pf_data:
             corr_matrix = pf_data["asset_returns"].corr()
             
             fig_corr = px.imshow(
                 corr_matrix,
                 text_auto=".2f",
                 aspect="auto",
                 color_continuous_scale=['#FBFAB8', '#1ED760', '#123524'], # Custom Theme Scale
                 zmin=-1,
                 zmax=1
             )
             fig_corr.update_layout(
                 margin=dict(l=0, r=0, t=0, b=0),
                 height=450
             )
             st.plotly_chart(fig_corr, use_container_width=True)
        else:
             st.info("No asset return data available for correlation.")

# ==========================================
# TAB 2: BENCHMARK COMPARISON
# ==========================================
with main_tabs[1]:
    st.markdown('<h3 style="margin-top: -10px;">Benchmark comparison</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Performance relative to selected benchmark
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Selection moved to top

    # Fetch Benchmark Data
    with st.spinner(f"Fetching {benchmark_ticker} data..."):
        bench_prices = load_price_data([benchmark_ticker], start_date, end_date)
        
        if bench_prices.empty:
            st.error(f"Could not load data for {benchmark_ticker}")
        else:
            # Compute daily returns for benchmark
            bench_returns = compute_returns(bench_prices)[benchmark_ticker]
            
            # Align Data (Inner Join)
            comparison_df = pd.DataFrame({
                "Portfolio": portfolio_returns,
                "Benchmark": bench_returns
            }).dropna()
            
            if comparison_df.empty:
                st.warning("No overlapping data between Portfolio and Benchmark.")
            else:
                p_ret = comparison_df["Portfolio"]
                b_ret = comparison_df["Benchmark"]
                
                # --- KPI CALCULATIONS (STRICT) ---
                
                # 1. Benchmark Stats
                b_cum_return = (1 + b_ret).cumprod().iloc[-1] - 1
                b_ann_return = b_ret.mean() * 252
                b_ann_vol = b_ret.std() * np.sqrt(252)
                
                # 2. Portfolio Stats (Recalculated on aligned data for consistency)
                p_ann_return = p_ret.mean() * 252
                
                # 3. Active Return (Difference of Annualized Returns)
                active_return = p_ann_return - b_ann_return
                
                # 4. Excess Return (Portfolio Annualized Return - Rf(0%))
                excess_return = p_ann_return # - 0.0
                
                # 5. Tracking Error (Std Dev of Active Daily Returns * sqrt(252))
                # Note: User request wrote "sigma * 252", assuming standard convention of sqrt(252) for annualized vol.
                diff_returns = p_ret - b_ret
                tracking_error = diff_returns.std() * np.sqrt(252)
                
                # 6. Information Ratio (Active Return / Tracking Error)
                information_ratio = active_return / tracking_error if tracking_error != 0 else 0.0
                
                # --- DISPLAY KPIS ---
                
                # Helper for metric card (redefined to ensure scope access)
                def bench_card(title, value, help_txt):
                     st.markdown(f"""
                        <div style="
                            border: 1px solid #7c7c7c;
                            padding: 15px;
                            border-radius: 0px;
                            margin-bottom: 10px;
                        ">
                            <div style="font-size: 16px; font-weight: 600; margin-bottom: 2px;">{title}</div>
                            <div style="font-size: 12px; opacity: 0.6; margin-bottom: 8px;">{help_txt}</div>
                            <div style="font-size: 28px; font-weight: 500;">{value}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Row 1: Benchmark Stats
                # Row 1: Benchmark Stats
                st.markdown("**Benchmark performance**")
                st.markdown("""
                <div style="margin-top: -15px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
                    Key metrics for the selected benchmark
                </div>
                """, unsafe_allow_html=True)
                row1 = st.columns(3)
                with row1[0]:
                    bench_card("Cumulative return", f"{b_cum_return:.2%}", "Total period growth")
                with row1[1]:
                    bench_card("Annualized return", f"{b_ann_return:.2%}", "Yearly average return")
                with row1[2]:
                    bench_card("Annualized volatility", f"{b_ann_vol:.2%}", "Annualized risk")
                
                # Row 2: Comparative Stats
                st.divider()

                st.markdown("#### Cumulative return vs Benchmark")
                st.markdown("""
                <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
                    Growth of $1 invested in Portfolio vs Benchmark
                </div>
                """, unsafe_allow_html=True)
                
                cum_df = (1 + comparison_df).cumprod()
                # reset index for plotly
                cum_df = cum_df.reset_index()
                # melt
                cum_melt = cum_df.melt(id_vars="Date", var_name="Asset", value_name="Cumulative return")
                
                fig_comp = px.line(cum_melt, x="Date", y="Cumulative return", color="Asset",
                                  color_discrete_map={"Portfolio": "#1ed760", "Benchmark": "#b3b3b3"})
                
                # Add fill with low opacity
                for trace in fig_comp.data:
                    trace.fill = "tozeroy"
                    if trace.line.color:
                        c = trace.line.color.lstrip('#')
                        rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
                        trace.fillcolor = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)"
                        
                fig_comp.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Growth of $1",
                    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                st.divider()
                st.markdown("**Relative Performance**")
                st.markdown("""
                <div style="margin-top: -15px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
                    Comparison of portfolio vs benchmark
                </div>
                """, unsafe_allow_html=True)
                row2 = st.columns(4)
                with row2[0]:
                    bench_card("Active return", f"{active_return:.2%}", "Port Ann. - Bench Ann.")
                with row2[1]:
                    bench_card("Excess return", f"{excess_return:.2%}", "Port Ann. - Rf (0%)")
                with row2[2]:
                    bench_card("Tracking error", f"{tracking_error:.2%}", "Vol. of return difference")
                with row2[3]:
                    bench_card("Information ratio", f"{information_ratio:.2f}", "Active Return / TE")
                
                st.divider()
                
                # --- CHARTS ---
                
                # Chart 1 moved to above Relative Performance
                
                st.markdown("#### Active return over time")
                st.markdown("""
                <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
                    Difference in cumulative returns (Portfolio - Benchmark)
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate difference of cumulative returns
                # (1 + Rp)_cum - (1 + Rb)_cum
                cum_port = (1 + p_ret).cumprod()
                cum_bench = (1 + b_ret).cumprod()
                active_cum_series = cum_port - cum_bench
                
                active_df = pd.DataFrame(active_cum_series, columns=["Active return"])
                
                fig_active = px.line(active_df, x=active_df.index, y="Active return")
                
                # Conditional Coloring based on final active return
                final_active = active_cum_series.iloc[-1]
                if final_active < 0:
                    act_color = "#ff4b4b"
                    act_fill = "rgba(255, 75, 75, 0.2)"
                else:
                    act_color = "#1ed760"
                    act_fill = "rgba(30, 215, 96, 0.2)"
                
                fig_active.update_traces(
                    line_color=act_color,
                    fill="tozeroy",
                    fillcolor=act_fill
                )
                
                # Add a zero line
                fig_active.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                
                fig_active.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=350,
                    xaxis_title="Date",
                    yaxis_title="Active return",
                    showlegend=False
                )
                st.plotly_chart(fig_active, use_container_width=True)

# ==========================================
# TAB 3: FACTOR MODELS
# ==========================================
with main_tabs[2]:
    st.markdown('<h3 style="margin-top: -10px;">Factor models</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Analyze portfolio performance using regression-based factor models
    </div>
    """, unsafe_allow_html=True)
    
    # Segmented Control for Model Selection
    model_choice = st.segmented_control(
        "Select model",
        options=["CAPM", "Fama-French 3-Factor"],
        selection_mode="single",
        default="CAPM"
    )
    

    
    # --- MODEL 1: CAPM ---
    if model_choice == "CAPM":
        st.markdown('<h4 style="margin-top: 0px;">Market risk decomposition</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
            Analysis of systematic risk (Beta) and skill (Alpha) relative to the benchmark
        </div>
        """, unsafe_allow_html=True)
        
        # helper styling for this tab
        def capm_card(title, value, help_txt):
             st.markdown(f"""
                <div style="
                    border: 1px solid #7c7c7c;
                    padding: 15px;
                    border-radius: 0px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 2px;">{title}</div>
                    <div style="font-size: 12px; opacity: 0.6; margin-bottom: 8px;">{help_txt}</div>
                    <div style="font-size: 28px; font-weight: 500;">{value}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with st.spinner("Running CAPM Regression..."):
            # Re-fetch/Align Data (Cached)
            bench_prices_capm = load_price_data([benchmark_ticker], start_date, end_date)
            
            if not bench_prices_capm.empty:
                 bench_ret_capm = compute_returns(bench_prices_capm)[benchmark_ticker]
                 
                 # Align
                 capm_df = pd.DataFrame({
                     "Portfolio": portfolio_returns,
                     "Benchmark": bench_ret_capm
                 }).dropna()
                 
                 if not capm_df.empty:
                     Y = capm_df["Portfolio"]
                     X = capm_df["Benchmark"]
                     
                     # Linear Regression (Polyfit degree 1)
                     beta, alpha_daily = np.polyfit(X, Y, 1)
                     
                     # Metrics
                     alpha_ann = (1 + alpha_daily)**252 - 1
                     r_squared = (Y.corr(X)) ** 2
                     
                     # Display Metrics
                     c1, c2, c3 = st.columns(3)
                     with c1:
                         capm_card("Portfolio beta", f"{beta:.2f}", "Sensitivity to market movements")
                     with c2:
                         capm_card("Annualized alpha", f"{alpha_ann:.2%}", "Excess return independent of market")
                     with c3:
                         capm_card("R-squared", f"{r_squared:.2f}", "Explained variance by market")
                     
                     st.divider()
                     
                     # Scatter Plot
                     st.markdown("#### Portfolio vs Market")
                     st.markdown(f"""
                     <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
                         CAPM Regression (Beta : {beta:.2f})
                     </div>
                     """, unsafe_allow_html=True)
                     
                     # Predicted line
                     X_seq = np.linspace(X.min(), X.max(), 100)
                     Y_seq = beta * X_seq + alpha_daily
                     
                     fig_capm = px.scatter(capm_df, x="Benchmark", y="Portfolio", opacity=0.6)
                     
                     # Add regression line
                     fig_capm.add_scatter(x=X_seq, y=Y_seq, mode='lines', name='Regression Line', 
                                        line=dict(color='#1ed760', width=3))
                                        
                     fig_capm.update_traces(marker=dict(color='#b3b3b3'))
                     fig_capm.update_layout(
                         margin=dict(l=0, r=0, t=30, b=0),
                         xaxis_title=f"Benchmark returns ({benchmark_ticker})",
                         yaxis_title="Portfolio returns",
                         height=450,
                         showlegend=True,
                         legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                     )
                     st.plotly_chart(fig_capm, use_container_width=True)
                     
                 else:
                     st.warning("Insufficient data overlap for CAPM.")
            else:
                st.error("Could not load benchmark data for CAPM.")

    # --- MODEL 2: FAMA-FRENCH ---
    elif model_choice == "Fama-French 3-Factor":
        st.markdown('<h4 style="margin-top: 0px;">Factor attribution</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
            Decomposition of returns into market, size, and value factors
        </div>
        """, unsafe_allow_html=True)
        
        # Reuse capm_card styling logic (redefined for safety or just reuse card style)
        def factor_card(title, value, help_txt):
             st.markdown(f"""
                <div style="
                    border: 1px solid #7c7c7c;
                    padding: 15px;
                    border-radius: 0px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 2px;">{title}</div>
                    <div style="font-size: 12px; opacity: 0.6; margin-bottom: 8px;">{help_txt}</div>
                    <div style="font-size: 28px; font-weight: 500;">{value}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("Fetching Fama-French Factors & Running Regression..."):
            # Fetch FF3 Data
            ff_data = get_fama_french_factors()
            
            if not ff_data.empty:
                # Align Data
                # Portfolio Returns are already computed as 'portfolio_returns' (Series)
                
                # Combine
                # FF Data columns: ['Mkt-RF', 'SMB', 'HML', 'RF']
                aligned_df = pd.DataFrame({"Port": portfolio_returns}).join(ff_data, how="inner").dropna()
                
                if not aligned_df.empty:
                    # Prepare Regression Variables
                    # Y = Portfolio Excess Return = Port - RF
                    Y = aligned_df["Port"] - aligned_df["RF"]
                    
                    # X = [Mkt-RF, SMB, HML] + Intercept
                    X = aligned_df[["Mkt-RF", "SMB", "HML"]]
                    
                    # Add constant for intercept (Alpha) manually for numpy
                    # We use np.linalg.lstsq
                    # Construct design matrix A = [1, Mkt-RF, SMB, HML]
                    A = np.column_stack([np.ones(len(X)), X["Mkt-RF"], X["SMB"], X["HML"]])
                    
                    # Run Regression
                    # output: coefs, residuals, rank, s
                    # coefs order: [Alpha, Beta_MKT, Beta_SMB, Beta_HML]
                    coeffs, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
                    
                    alpha_daily = coeffs[0]
                    beta_mkt = coeffs[1]
                    beta_smb = coeffs[2]
                    beta_hml = coeffs[3]
                    
                    # Calculate R-squared
                    y_pred = A @ coeffs
                    ss_res = np.sum((Y - y_pred)**2)
                    ss_tot = np.sum((Y - Y.mean())**2)
                    r_squared_ff = 1 - (ss_res / ss_tot)
                    
                    # Annualize Alpha
                    alpha_ann_ff = (1 + alpha_daily)**252 - 1
                    
                    # Display Metrics
                    # Row 1: All 5 KPIs Side-by-Side
                    row1 = st.columns(5)
                    with row1[0]:
                        factor_card("Market beta", f"{beta_mkt:.2f}", "Sensitivity to Market")
                    with row1[1]:
                        factor_card("Size beta", f"{beta_smb:.2f}", "Small Cap Exposure")
                    with row1[2]:
                        factor_card("Value beta", f"{beta_hml:.2f}", "Value Exposure")
                    with row1[3]:
                        factor_card("Annualized alpha", f"{alpha_ann_ff:.2%}", "Unexplained Return")
                    with row1[4]:
                        factor_card("R-squared", f"{r_squared_ff:.2f}", "Explained Variance")
                    
                    st.divider()
                    
                    # Chart: Factor Exposures
                    st.markdown("#### Factor exposures")
                    st.markdown("""
                    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
                        Fama-French 3-Factor Betas
                    </div>
                    """, unsafe_allow_html=True)
                    
                    factors_df = pd.DataFrame({
                        "Factor": ["Market (MKT)", "Size (SMB)", "Value (HML)"],
                        "Beta": [beta_mkt, beta_smb, beta_hml]
                    })
                    
                    fig_ff = px.bar(factors_df, x="Factor", y="Beta", text_auto=".2f")
                    
                    fig_ff.update_traces(
                        marker_color='#1ed760',
                        textfont_color='white',
                        textposition='inside',
                        insidetextanchor='middle'
                    )
                    fig_ff.update_layout(
                        margin=dict(l=0, r=0, t=30, b=0),
                        yaxis_title="Beta coefficient",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_ff, use_container_width=True)
                    
                else:
                     st.warning("Insufficient overlapping data for Fama-French model.")
            else:
                st.error("Could not fetch Fama-French data.")
