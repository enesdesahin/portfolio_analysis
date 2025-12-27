import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils.data_loader import load_price_data
from utils.returns import compute_returns
from utils.risk import calculate_historical_var, calculate_historical_cvar, simulate_monte_carlo
from utils.social import render_social_links



st.title("Risk analysis")
st.markdown("""
<div style="margin-top: -10px; opacity: 0.6; margin-bottom: 30px;">
    Advanced risk analysis : Tail events, historical crisis scenarios, and future uncertainty
</div>
""", unsafe_allow_html=True)

render_social_links(clean_layout=True)

# 0. Data Loading (Reused from session or cache)
if "portfolio" not in st.session_state or not st.session_state["portfolio"]:
    st.warning("Please build a portfolio first", icon=":material/error:")
    st.stop()
    
pf_data = st.session_state["portfolio"]
tickers = pf_data["tickers"]
weights = pf_data["weights"]
start_date = pf_data["start_date"]
end_date = pf_data["end_date"]

# Compute Portfolio Returns
prices = load_price_data(tickers, start_date, end_date)
asset_returns = compute_returns(prices)
# Weighted sum for portfolio return
portfolio_returns = asset_returns.dot(weights)

# Helper for tabs
risk_tabs = st.tabs(["Historical tail risk", "Stress scenarios", "Future simulation"])

# ==========================================
# LAYER 1: HISTORICAL TAIL RISK
# ==========================================
with risk_tabs[0]:
    st.markdown("### Historical tail risk analysis")
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Based on realized daily returns distribution (no parametric assumptions)
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    var_95 = calculate_historical_var(portfolio_returns, 0.95)
    var_99 = calculate_historical_var(portfolio_returns, 0.99)
    cvar_95 = calculate_historical_cvar(portfolio_returns, 0.95)
    
    # Helper for custom metric card (Consistent with Analytics)
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

    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("VaR 95%", "Max loss with 95% confidence (1 in 20 days)", f"-{var_95:.2%}")
    with col2:
        kpi_card("VaR 99%", "Max loss with 99% confidence (1 in 100 days)", f"-{var_99:.2%}")
    with col3:
        kpi_card("CVaR 95%", "Average loss in the worst 5% of days", f"-{cvar_95:.2%}")
    
    # Histogram Visualization
    with st.container(border=True):
        st.markdown("""
        <div style="margin-bottom: 5px;">
            <div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">Distribution of daily returns</div>
            <div style="font-size: 13px; opacity: 0.6;">Daily return distribution & risk cutoffs</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Actually we want histogram of VALUES, not time series
        fig_hist = px.histogram(x=portfolio_returns, nbins=100, labels={'x': 'Daily return'})
        
        fig_hist.update_traces(marker_color='#1ed760', marker_line_width=1, marker_line_color="rgba(0,0,0,0.4)")
        
        # Add Vertical Lines for VaR
        # VaR is positive loss, return is negative
        var_95_ret = -var_95
        var_99_ret = -var_99
        cvar_95_ret = -cvar_95
        
        # Add Vertical Lines for VaR with Staggered Labels
        # Colors: VaR 95 (Amber), CVaR 95 (Red), VaR 99 (Dark Red)
        
        # 1. VaR 95% - Amber, Solid
        fig_hist.add_vline(x=var_95_ret, line_width=2, line_dash="solid", line_color="#FFC107")
        fig_hist.add_annotation(
            x=var_95_ret, y=0.95, yref="paper",
            text=f"VaR 95%",
            showarrow=False,
            font=dict(color="#FFC107", size=12, weight="bold"),
            xanchor="left", xshift=5
        )
        
        # 2. CVaR 95% - Red, Dash
        fig_hist.add_vline(x=cvar_95_ret, line_width=2, line_dash="dash", line_color="#FF5252")
        fig_hist.add_annotation(
            x=cvar_95_ret, y=0.85, yref="paper", # Staggered lower
            text=f"CVaR 95%",
            showarrow=False,
            font=dict(color="#FF5252", size=12, weight="bold"),
            xanchor="left", xshift=5
        )
        
        # 3. VaR 99% - Dark Red, Dot
        fig_hist.add_vline(x=var_99_ret, line_width=2, line_dash="dot", line_color="#D32F2F")
        fig_hist.add_annotation(
            x=var_99_ret, y=0.75, yref="paper", # Staggered even lower
            text=f"VaR 99%",
            showarrow=False,
            font=dict(color="#D32F2F", size=12, weight="bold"),
            xanchor="left", xshift=5
        )
        
        fig_hist.update_layout(
            xaxis_tickformat='.1%',
            showlegend=False,
            height=400,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# LAYER 2: HISTORICAL STRESS TESTING
# ==========================================
with risk_tabs[1]:
    st.markdown("### Stress scenarios")
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        How the portfolio would have performed during historical market crashes
    </div>
    """, unsafe_allow_html=True)
    
    # Define Scenarios
    scenarios = {
        "2007–2009 (Subprime crisis)": ("2007-01-01", "2009-12-31"),
        "2020 (COVID shock)": ("2020-01-01", "2020-12-31"),
        "2022 (Inflation & rates shock)": ("2022-01-01", "2022-12-31")
    }
    
    col_scenario, _ = st.columns([1, 3])
    with col_scenario:
        selected_scenario = st.selectbox("Select crisis scenario", list(scenarios.keys()))
    start_s, end_s = scenarios[selected_scenario]
    
    # Filter Data (Need to reload/filter for specific dates if not in current range)
    # Our 'asset_returns' only covers user selected range.
    # To do this properly, we need to fetch data for these specific dates.
    # We use load_price_data which caches.
    
    with st.spinner(f"Simulating {selected_scenario}..."):
        try:
            # Fetch scenario data
            s_prices = load_price_data(tickers, start_s, end_s)
            if s_prices.empty:
                st.error("No data available for this period (assets might not have existed).")
            else:
                s_returns = compute_returns(s_prices)
                
                # Align weights (re-normalize if assets missing? No, assume fixed portfolio)
                # If asset didn't exist, we can't simulate.
                # Drop assets with no data?
                valid_tickers = [t for t in tickers if t in s_returns.columns]
                
                if not valid_tickers:
                    st.error("None of the portfolio assets have data available for this historical period.")
                else:
                    if len(valid_tickers) < len(tickers):
                        st.warning(f"Some assets missing data for this period: {set(tickers) - set(valid_tickers)}")
                        
                    # Re-weight? Or just use available? 
                    # Strict approach: If asset didn't exist, we can't backtest it in that form.
                    # Proxy: Re-normalize weights of available assets.
                    s_weights = np.array([weights[tickers.index(t)] for t in valid_tickers])
                    
                    if s_weights.sum() == 0:
                         st.error("Remaining assets have zero weight.")
                    else:
                        s_weights = s_weights / s_weights.sum()
                
                        s_weights = s_weights / s_weights.sum()
                
                        s_port_ret = s_returns[valid_tickers].dot(s_weights)
                        
                        if s_port_ret.empty:
                             st.error("Insufficient data to run simulation for this period", icon=":material/error:")
                        else:
                            # Metrics
                            cum_ret = (1 + s_port_ret).cumprod() - 1
                            
                            if len(cum_ret) > 0:
                                total_return = cum_ret.iloc[-1]
                                
                                # Max Drawdown
                                roll_max = (1 + s_port_ret).cumprod().cummax()
                                drawdown = (1 + s_port_ret).cumprod() / roll_max - 1
                                max_dd = drawdown.min()
                                
                                col_kpi1, col_kpi2 = st.columns(2)
                                with col_kpi1:
                                    kpi_card("Period return", "Total return during crisis", f"{total_return:.2%}")
                                with col_kpi2:
                                    kpi_card("Max drawdown", "Maximum peak-to-trough decline", f"{max_dd:.2%}")
                                
                                # Chart with consistent styling
                                with st.container(border=True):
                                    st.markdown("""
                                    <div style="margin-bottom: 5px;">
                                        <div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">Cumulative performance</div>
                                        <div style="font-size: 13px; opacity: 0.6;">Portfolio wealth progression during the crisis</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Wealth Index
                                    s_cum_df = (1 + s_port_ret).cumprod()
                                    
                                    # Determine color based on final return
                                    # If total_return < 0 -> Red, else Green
                                    if total_return >= 0:
                                        line_color = "#26de81"
                                        fill_color = "rgba(38, 222, 129, 0.2)"
                                    else:
                                        line_color = "#ff4b4b"
                                        fill_color = "rgba(255, 75, 75, 0.2)"
                                    
                                    fig_stress = go.Figure()
                                    fig_stress.add_trace(go.Scatter(
                                        x=s_cum_df.index,
                                        y=s_cum_df.values,
                                        mode='lines',
                                        line=dict(color=line_color, width=2),
                                        fill='tozeroy',
                                        fillcolor=fill_color,
                                        name='Portfolio Value'
                                    ))
                                    
                                    fig_stress.update_layout(
                                        margin=dict(l=0, r=0, t=10, b=0),
                                        height=350,
                                        showlegend=False,
                                        yaxis_title="Wealth Index (Start=1.0)",
                                        template="plotly_dark",
                                        xaxis=dict(showgrid=False),
                                        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)'
                                    )
                                    st.plotly_chart(fig_stress, use_container_width=True)
                            else:
                               st.error("Insufficient data points for this period.", icon=":material/error:") 
                
        except Exception as e:
            st.error(f"Error simulating scenario: {e}")

# ==========================================
# LAYER 3: MONTE CARLO SIMULATION
# ==========================================
with risk_tabs[2]:
    st.markdown("### Monte-Carlo simulation")
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Projected future returns using historical mean and volatility (assumes normal distribution)
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("This is a statistical simulation, not a forecast. Past performance is not indicative of future results.", icon=":material/error:")
    
    # Inputs
    days_proj = st.slider("Projection days", 30, 252, 125)
    num_sims = 100
    
    if st.button("Run simulation"):
        with st.spinner("Running 100 simulations..."):
            # Calculate Portfolio Mean & Vol
            daily_mean = portfolio_returns.mean()
            daily_vol = portfolio_returns.std()
            
            # Run simulation
            sim_paths = simulate_monte_carlo(daily_mean, daily_vol, days_proj, num_sims)
            
            # Visualize "Cone of Uncertainty"
            # We want to show Median, 5th, and 95th percentiles paths
            
            # Calculate percentiles across simulations for each day
            # shape: (sims, days+1)
            median_path = np.percentile(sim_paths, 50, axis=0)
            p95_path = np.percentile(sim_paths, 95, axis=0)
            p05_path = np.percentile(sim_paths, 5, axis=0)
            
            # Create Chart
            x_axis = np.arange(days_proj + 1)
            
            # --- KPIs (Moved Above Chart) ---
            col_mc1, col_mc2, col_mc3 = st.columns(3)
            with col_mc1:
                kpi_card("Median outcome", "Expected portfolio value", f"{median_path[-1]:.2f}x")
            with col_mc2:
                kpi_card("Upside (95%)", "Optimistic scenario", f"{p95_path[-1]:.2f}x")
            with col_mc3:
                kpi_card("Downside (5%)", "Pessimistic scenario", f"{p05_path[-1]:.2f}x")
            
            # --- Chart ---
            with st.container(border=True):
                st.markdown(f"""
                <div style="margin-bottom: 5px;">
                    <div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">Projected growth (next {days_proj} days)</div>
                    <div style="font-size: 13px; opacity: 0.6;">100 simulated paths (Monte-Carlo)</div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_mc = go.Figure()
                
                # Spaghetti Plot: All paths with low opacity
                # Limit to first 100 if num_sims is large to avoid lag
                for i in range(min(len(sim_paths), 200)):
                    fig_mc.add_trace(go.Scatter(
                        x=x_axis, y=sim_paths[i],
                        mode='lines',
                        line=dict(color='#1ed760', width=1),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Median Path (Highlight)
                fig_mc.add_trace(go.Scatter(
                    x=x_axis, y=median_path,
                    mode='lines', line=dict(color='#FFFFFF', width=2),
                    name='Median Path',
                    opacity=1.0
                ))
                
                fig_mc.update_layout(
                    xaxis_title="Days",
                    yaxis_title="Portfolio value",
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", y=1.02),
                    template="plotly_dark",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
