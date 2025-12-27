import streamlit as st
from utils.social import render_social_links

# Global config must be set first
st.set_page_config(
    page_title="Portfolio Analytics",
    layout="wide"
)

# Global CSS: Rectangular buttons & Consistent Top Spacing
st.markdown("""
<style>
div.stButton > button:first-child {
    border-radius: 0px;
}
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}
h1 {
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Define pages using existing files
# Note: st.Page takes the file path relative to the main script
pages = [
    st.Page("home.py", 
            title="Home", 
            icon=":material/home:"
    ),
    st.Page("pages/portfolio_builder.py", 
            title="Builder", 
            icon=":material/widgets:"
    ),
    st.Page("pages/portfolio_analytics.py", 
            title="Analytics", 
            icon=":material/insert_chart:"
    ),
    st.Page("pages/portfolio_optimization.py", 
            title="Optimization", 
            icon=":material/donut_small:"
    ),
    st.Page("pages/tail_risk.py", 
            title="Risk analysis", 
            icon=":material/earthquake:"
    )
]

# Create navigation
pg = st.navigation(pages)

# Run the selected page
pg.run()


