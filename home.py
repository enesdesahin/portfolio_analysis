
import streamlit as st
from utils.social import get_social_links_html

# Custom Sidebar Card (Visual match for native style)
with st.sidebar:
    social_html = get_social_links_html(show_attribution=False)
    footer_section = ""
    if social_html:
        # Wrapper with margin-top to separate from description
        footer_section = f'<div style="margin-top: 16px;">{social_html}</div>'

    card_template = """
    <div style="
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    ">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span style="font-weight: 600; font-size: 16px; color: #e8eaed;">Contact</span>
        </div>
        <div style="font-size: 13px; color: rgba(248, 250, 252, 0.7); line-height: 1.4;">
            Actively seeking an internship in portfolio management starting May 2026.
        </div>
        {{FOOTER}}
    </div>
    """
    
    st.markdown(card_template.replace("{{FOOTER}}", footer_section), unsafe_allow_html=True)

# Custom CSS for rectangular buttons (matching portfolio_builder.py)
st.markdown("""
<style>
div.stButton > button:first-child {
    border-radius: 0px;
}
</style>
""", unsafe_allow_html=True)

# Layout: 1/6 spacer, 4/6 content, 1/6 spacer (approx centering)
# Adjust ratio as needed for "900-1100px" look on wide screens.
# [1, 2, 1] on wide screen (ratio 0.25, 0.5, 0.25) -> 50% width.
_, col_center, _ = st.columns([1, 2, 1])

with col_center:
    st.title("Portfolio analysis", anchor=False)

    st.markdown("""
    <div style="text-align: justify; margin-bottom: -20px;">
    This application is designed to support the full investment lifecycle of a multi-asset portfolio. It follows a structured, sequential workflow where each analytical module builds upon the decisions made in the previous steps. Users are expected to proceed through the stages in order to ensure data consistency and logical progression.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<h3 style="margin-top: -20px;">Methodological workflow</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: left; font-size: 1rem; opacity: 0.7; margin-bottom: 1rem; margin-top: -5px;">
    End-to-end portfolio construction, performance analysis, and risk-aware decision support
    </div>
    """, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        **:material/widgets: Builder**

        Build a customized investment portfolio by selecting assets from a global universe and defining their allocation weights.

        **:material/insert_chart: Analytics**

        Conduct a rigorous analysis of historical performance, compare against benchmarks, and decompose returns using CAPM and Fama-French factor models.

        **:material/donut_small: Optimization**

        Apply modern portfolio theory (MVO) to mathematically reallocate weights to maximize the Sharpe ratio under specific constraints.

        **:material/earthquake: Risk analysis**

        Assess downside risk using VaR, CVaR, historical crisis scenarios (2008, 2020, 2022), and future uncertainty via Monte-Carlo simulations.
        """)

    st.markdown('<hr style="margin-top: 20px;">', unsafe_allow_html=True)
    if st.button("Get started"):
        st.switch_page("pages/portfolio_builder.py")
