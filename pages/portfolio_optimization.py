import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.optimization import run_mean_variance_optimization, get_efficient_frontier
from utils.returns import compute_returns
from utils.data_loader import load_price_data
from utils.risk import calculate_beta
from utils.social import render_social_links

st.title("Optimization")
st.markdown("""
<div style="margin-top: -10px; opacity: 0.6; margin-bottom: -20px;">
    Reallocate weights using mean-variance optimization (MVO)
</div>
""", unsafe_allow_html=True)

st.divider()

render_social_links(clean_layout=True)

# 1. Precondition Check
if "portfolio" not in st.session_state or not st.session_state["portfolio"]:
    st.warning("Please build a portfolio first", icon=":material/error:")
    st.stop()
    
pf_data = st.session_state["portfolio"]
tickers = pf_data["tickers"]
current_weights_list = pf_data["weights"]
current_weights = dict(zip(tickers, current_weights_list))

# Load Data (Do NOT reload, use cache or fetch same range)
# But utils needs data. We use load_price_data which is cached.
start_date = pf_data["start_date"]
end_date = pf_data["end_date"]
benchmark_ticker = "^GSPC" # Default for Beta constraint

# Prepare Data
with st.spinner("Preparing data for optimization..."):
    # Load Asset Prices
    prices = load_price_data(tickers, start_date, end_date)
    asset_returns = compute_returns(prices)
    
    # Load Benchmark for Beta Constraint
    bench_prices = load_price_data([benchmark_ticker], start_date, end_date)
    bench_returns = compute_returns(bench_prices)[benchmark_ticker]
    
    # Align Data
    # Inner join assets and benchmark
    combined_df = asset_returns.join(bench_returns, how="inner").dropna()
    
    if combined_df.empty:
        st.error("Insufficient data overlap between assets and benchmark.")
        st.stop()
        
    aligned_assets = combined_df[tickers]
    aligned_bench = combined_df[benchmark_ticker]
    
    # Calculate Current Beta (for constraint reference)
    # Portfolio Return = sum(w * R)
    curr_port_ret = aligned_assets.dot(list(current_weights.values()))
    
    # Beta = Cov(Rp, Rb) / Var(Rb)
    curr_beta = curr_port_ret.cov(aligned_bench) / aligned_bench.var()

# UI Layout
st.markdown('<h3 style="margin-top: -20px;">Settings</h3>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
    Define your optimization constraints and objectives
</div>
""", unsafe_allow_html=True)
col_sett1, col_sett2 = st.columns(2)
with col_sett1:
    st.success(f"**Current portfolio beta :** {curr_beta:.2f}  \nTarget beta range : +/- 0.2")

with col_sett2:
    st.success("**Objective :** Maximize sharpe ratio  \nConstraints : Long-only, fully-invested, max 40% per asset")

if st.button("Optimize portfolio"):
    with st.spinner("Running Mean-Variance Optimization..."):
        
        # Run MVO
        opt_weights_dict, opt_beta = run_mean_variance_optimization(
            current_weights, 
            aligned_assets, 
            aligned_bench, 
            max_weight=0.40,
            beta_target=curr_beta,
            beta_tol=0.2
        )
        
        if opt_weights_dict is None:
            st.error("Optimization failed to find a solution satisfying all constraints.")
            st.warning("Try relaxing constraints or checking data.")
        else:
            # Store Result
            st.session_state["optimized_portfolio"] = opt_weights_dict
            st.toast("Optimization successful !")
            
            # --- RESULTS SECTION ---
            st.divider()
            
            # 1. Comparison Table
            st.markdown("#### Weight allocation")
            st.markdown("""
            <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
                Comparison between current and optimized portfolio weights
            </div>
            """, unsafe_allow_html=True)
            
            comp_df = pd.DataFrame({
                "Asset": tickers,
                "Current weight": [current_weights[t] for t in tickers],
                "Optimized weight": [opt_weights_dict[t] for t in tickers]
            })
            comp_df["Change"] = comp_df["Optimized weight"] - comp_df["Current weight"]
            
            # Merge Metadata if available
            if "metadata" in pf_data and pf_data["metadata"] is not None:
                meta_df = pf_data["metadata"]
                comp_df = comp_df.merge(meta_df, left_on="Asset", right_index=True, how="left")
            
            # Rename Asset -> Ticker
            comp_df.rename(columns={"Asset": "Ticker"}, inplace=True)
            
            # Reorder columns
            desired_order = ["Company Name", "Ticker"]
            weight_cols = ["Current weight", "Optimized weight", "Change"]
            base_cols = [c for c in comp_df.columns if c not in desired_order and c not in weight_cols]
            
            final_cols = desired_order + base_cols + weight_cols
            # Filter existing
            final_cols = [c for c in final_cols if c in comp_df.columns]
            comp_df = comp_df[final_cols]
            
            # Custom Styling for Change column
            # Institutional Theme: Muted text colors with subtle background
            def style_change_col(v):
                if pd.isna(v): return ""
                
                # Colors
                pos_color = "#5CAB7D"
                pos_bg = "rgba(92, 171, 125, 0.12)" # ~12% opacity
                
                neg_color = "#B65C5C" 
                neg_bg = "rgba(182, 92, 92, 0.12)" # ~12% opacity
                
                neu_color = "#B0B0B0"
                
                style = "font-weight: 500;"
                
                if v > 0.0001: # Epsilon for float zero
                    style += f"color: {pos_color}; background-color: {pos_bg};"
                elif v < -0.0001:
                    style += f"color: {neg_color}; background-color: {neg_bg};"
                else:
                    style += f"color: {neu_color};"
                    
                return style

            st.dataframe(comp_df.style.format({
                "Current weight": "{:.2%}",
                "Optimized weight": "{:.2%}",
                "Change": "{:+.2%}"
            }).map(style_change_col, subset=["Change"]), hide_index=True, use_container_width=True)
            
            # 2. KPI Comparison
            st.markdown("#### Expected performance")
            st.markdown("""
            <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
                Comparison of key risk and return metrics based on historical data
            </div>
            """, unsafe_allow_html=True)
            
            # Helper metrics calc
            def calc_metrics(w_dict, rets):
                w_vec = np.array([w_dict[t] for t in tickers])
                p_r = rets.dot(w_vec)
                ann_ret = p_r.mean() * 252
                ann_vol = p_r.std() * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                return ann_ret, ann_vol, sharpe
            
            curr_ret, curr_vol, curr_sharpe = calc_metrics(current_weights, aligned_assets)
            opt_ret, opt_vol, opt_sharpe = calc_metrics(opt_weights_dict, aligned_assets)
            
            # Helper custom card (from analytics)
            def kpi_card(title, value, delta_val=None, inverse=False):
                if delta_val is not None:
                    # Determine color and arrow
                    is_positive = delta_val > 0
                    
                    # Logic: Green if Good.
                    if inverse:
                        color = "#ff4b4b" if is_positive else "#1ed760"
                    else:
                        color = "#1ed760" if is_positive else "#ff4b4b"
                        
                    arrow = "▲" if is_positive else "▼"
                    delta_str = f"{arrow} {abs(delta_val):.2%}"
                    
                    # Subtitle with colored delta text
                    subtitle_html = f'<span style="color: {color}; font-weight: bold;">{delta_str}</span>'
                else:
                    subtitle_html = ""

                st.markdown(f"""
                <div style="
                    border: 1px solid #7c7c7c;
                    padding: 15px;
                    border-radius: 0px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 2px;">{title}</div>
                    <div style="font-size: 12px; opacity: 0.6; margin-bottom: 8px;">{subtitle_html}</div>
                    <div style="font-size: 28px; font-weight: 500;">{value}</div>
                </div>
                """, unsafe_allow_html=True)
            
            kpi_cols = st.columns(3)
            with kpi_cols[0]:
                kpi_card("Expected return", f"{opt_ret:.2%}", delta_val=opt_ret-curr_ret)
            with kpi_cols[1]:
                kpi_card("Annualized volatility", f"{opt_vol:.2%}", delta_val=opt_vol-curr_vol, inverse=True)
            with kpi_cols[2]:
                kpi_card("Sharpe ratio", f"{opt_sharpe:.2f}", delta_val=opt_sharpe-curr_sharpe)
            
            # Vertical Spacer
            st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)
            
            st.info(f"**Optimized beta :** {opt_beta:.2f} (Target: {curr_beta-0.2:.2f} - {curr_beta+0.2:.2f})")
            
            # Vertical Spacer (Negative to pull closer)
            st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True)
            
            # 3. Visualizations
            
            # A. Efficient Frontier
            with st.container(border=True):
                st.markdown("""
                <div style="margin-bottom: 5px;">
                    <div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">Efficient frontier</div>
                    <div style="font-size: 13px; opacity: 0.6;">Risk vs return tradeoff analysis</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate Frontier (computationally expensive, maybe move to utils if slow)
                with st.spinner("Generating Efficient Frontier..."):
                    frontier_vols, frontier_rets = get_efficient_frontier(aligned_assets)
                
                fig_ef = go.Figure()
                
                # Frontier Line
                fig_ef.add_trace(go.Scatter(x=frontier_vols, y=frontier_rets, mode='lines', name='Efficient Frontier',
                                          line=dict(color='#1ed760', width=2)))
                
                # Current Portfolio
                fig_ef.add_trace(go.Scatter(x=[curr_vol], y=[curr_ret], mode='markers', name='Current Portfolio',
                                          marker=dict(color='#b3b3b3', size=10, symbol='x')))
                                          
                # Optimized Portfolio
                fig_ef.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', name='Optimized Portfolio',
                                          marker=dict(color='#1ed760', size=12, symbol='star')))
                
                fig_ef.update_layout(
                    xaxis_title="Annualized Volatility",
                    yaxis_title="Expected Return",
                    margin=dict(t=10, b=0, l=0, r=0),
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_ef, use_container_width=True)

            # B. Weight Comparison Chart
            with st.container(border=True):
                st.markdown("""
                <div style="margin-bottom: 5px;">
                    <div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">Allocation shift</div>
                    <div style="font-size: 13px; opacity: 0.6;">Comparison of current vs optimized weights</div>
                </div>
                """, unsafe_allow_html=True)
                
                df_melt = comp_df.melt(id_vars="Ticker", value_vars=["Current weight", "Optimized weight"], var_name="Type", value_name="Weight")
                fig_bar = px.bar(df_melt, x="Ticker", y="Weight", color="Type", barmode="group",
                                color_discrete_map={"Current weight": "#b3b3b3", "Optimized weight": "#1ed760"})
                fig_bar.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=350, legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_bar, use_container_width=True)
