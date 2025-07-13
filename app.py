import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Data Loading and Preprocessing
# -----------------------------
@st.cache_data

def load_data():
    try:
        df = pd.read_excel('Data.xlsx')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    # Data quality checks
    expected_cols = ['district', 'month', 'fiscal_year', 'sales_volume', 'temp_mean', 'wind_mean', 'gdd_cumulative',
                     'temp_zscore', 'wind_zscore', 'gdd_zscore', 'rainfall_avg', 'ndvi_mean', 'ndvi_zscore']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return None
    # Handle missing values
    df = df.dropna(subset=['district', 'fiscal_year', 'sales_volume'])
    # Clean fiscal_year: extract digits and convert to int
    df['fiscal_year'] = df['fiscal_year'].astype(str).str.extract(r'(\d+)')[0].astype(int)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

# -----------------------------
# 2. Correlation Calculation
# -----------------------------
def calculate_correlations(df, groupby_cols=['district', 'fiscal_year']):
    env_factors = ['temp_mean', 'wind_mean', 'gdd_cumulative', 'temp_zscore', 'wind_zscore', 'gdd_zscore',
                   'rainfall_avg', 'ndvi_mean', 'ndvi_zscore']
    results = []
    grouped = df.groupby(groupby_cols)
    for name, group in grouped:
        for var in env_factors:
            if group['sales_volume'].isnull().all() or group[var].isnull().all():
                continue
            try:
                r, p = pearsonr(group['sales_volume'], group[var])
            except Exception:
                r, p = np.nan, np.nan
            # Strength categorization
            abs_r = abs(r)
            if abs_r < 0.3:
                strength = 'Weak'
            elif abs_r < 0.7:
                strength = 'Moderate'
            else:
                strength = 'Strong'
            results.append({
                **{col: val for col, val in zip(groupby_cols, name if isinstance(name, tuple) else (name,))},
                'variable': var,
                'correlation': r,
                'p_value': p,
                'strength': strength
            })
    corr_df = pd.DataFrame(results)
    return corr_df

# -----------------------------
# 3. Correlation Heatmap
# -----------------------------
def create_correlation_heatmap(corr_df, district=None, fiscal_year=None):
    data = corr_df.copy()
    if district:
        data = data[data['district'] == district]
    if fiscal_year:
        data = data[data['fiscal_year'] == fiscal_year]
    heatmap_data = data.pivot(index='variable', columns='fiscal_year' if district else 'district', values='correlation')
    fig = px.imshow(heatmap_data, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                    labels=dict(color='Pearson r'), aspect='auto')
    fig.update_layout(title=f"Correlation Heatmap {'for ' + district if district else ''}",
                     xaxis_title='Fiscal Year' if district else 'District',
                     yaxis_title='Environmental Factor')
    return fig

# -----------------------------
# 4. Scatter Plots with Regression
# -----------------------------
def create_scatter_plots(df, district, fiscal_year, variable):
    data = df.copy()
    if district:
        data = data[data['district'] == district]
    if fiscal_year:
        data = data[data['fiscal_year'] == fiscal_year]
    fig = px.scatter(data, x=variable, y='sales_volume', color='month', trendline='ols',
                     title=f"Sales vs {variable} ({district}, {fiscal_year})", labels={'sales_volume': 'Sales Volume'})
    return fig

# -----------------------------
# 5. Box Plots
# -----------------------------
def create_box_plots(df, by='district'):
    fig = px.box(df, x=by, y='sales_volume', color=by,
                 title=f"Sales Volume Distribution by {by.capitalize()}",
                 labels={'sales_volume': 'Sales Volume'})
    return fig

# -----------------------------
# 6. Time Series Analysis
# -----------------------------
def create_time_series(df, district=None, fiscal_year=None):
    data = df.copy()
    if district:
        data = data[data['district'] == district]
    if fiscal_year:
        data = data[data['fiscal_year'] == fiscal_year]
    fig = px.line(data, x='month', y='sales_volume', color='fiscal_year', markers=True,
                  title=f"Monthly Sales Trends {'for ' + district if district else ''}",
                  labels={'sales_volume': 'Sales Volume'})
    return fig

# -----------------------------
# 7. Grouped Bar Charts (Average Correlations)
# -----------------------------
def create_grouped_bar_chart(corr_df):
    avg_corr = corr_df.groupby(['district', 'variable'])['correlation'].mean().reset_index()
    fig = px.bar(avg_corr, x='district', y='correlation', color='variable', barmode='group',
                 title='Average Correlations by District')
    return fig

# -----------------------------
# 8. Faceted Scatter Plots
# -----------------------------
def create_faceted_scatter(df, variable):
    fig = px.scatter(df, x=variable, y='sales_volume', color='district', facet_col='fiscal_year',
                     trendline='ols',
                     title=f"Sales vs {variable} Across Fiscal Years", labels={'sales_volume': 'Sales Volume'})
    return fig

# -----------------------------
# 9. Statistical Insights
# -----------------------------
def generate_insights(corr_df):
    insights = []
    for district in corr_df['district'].unique():
        sub = corr_df[corr_df['district'] == district]
        top = sub.loc[sub['correlation'].abs().idxmax()]
        sig = sub[sub['p_value'] < 0.05]
        insights.append(f"**{district}:** Top influencer: {top['variable']} (r={top['correlation']:.2f}, p={top['p_value']:.3f}, {top['strength']})")
        if not sig.empty:
            for _, row in sig.iterrows():
                insights.append(f"  - Significant: {row['variable']} (r={row['correlation']:.2f}, p={row['p_value']:.3f}, {row['strength']})")
    return '\n'.join(insights)

# -----------------------------
# 10. Export Results
# -----------------------------
def export_results(df, filename='correlation_results.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# -----------------------------
# 11. Streamlit App Layout
# -----------------------------
def main():
    st.set_page_config(page_title="District Sales-Environment Correlation Dashboard", layout="wide")
    st.title("📊 District-wise Sales & Environmental Correlation Analysis")
    st.markdown("""
    This dashboard analyzes the relationship between sales volume and environmental factors across districts and fiscal years.\
    Use the sidebar to filter and explore interactive visualizations and statistical insights.
    """)

    # Sidebar Filters
    df = load_data()
    if df is None or df.empty:
        st.stop()
    districts = sorted(df['district'].unique())
    fiscal_years = sorted(df['fiscal_year'].unique())
    env_vars = ['temp_mean', 'wind_mean', 'gdd_cumulative', 'temp_zscore', 'wind_zscore', 'gdd_zscore',
                'rainfall_avg', 'ndvi_mean', 'ndvi_zscore']
    st.sidebar.header("Filters")
    selected_district = st.sidebar.multiselect("Select District(s)", districts, default=districts)
    year_range = st.sidebar.slider("Fiscal Year Range", min_value=int(min(fiscal_years)), max_value=int(max(fiscal_years)),
                                   value=(int(min(fiscal_years)), int(max(fiscal_years))))
    selected_vars = st.sidebar.multiselect("Select Environmental Variable(s)", env_vars, default=env_vars)

    # Filter data
    df_filtered = df[df['district'].isin(selected_district) &
                     df['fiscal_year'].between(year_range[0], year_range[1])]

    # Calculate correlations
    corr_df = calculate_correlations(df_filtered)
    corr_df = corr_df[corr_df['variable'].isin(selected_vars)]

    # Tabs
    tabs = st.tabs(["Data Overview", "Correlation Analysis", "District Comparison", "Time Series Analysis", "Statistical Summary", "Export Results"])

    # Data Overview
    with tabs[0]:
        st.subheader("Data Overview")
        st.dataframe(df_filtered.head(100))
        st.markdown(f"**Rows:** {len(df_filtered)} | **Columns:** {len(df_filtered.columns)}")
        st.write("Missing values per column:")
        st.write(df_filtered.isnull().sum())

    # Correlation Analysis
    with tabs[1]:
        st.subheader("Correlation Heatmap (Overall)")
        st.plotly_chart(create_correlation_heatmap(corr_df), use_container_width=True)
        st.subheader("District-wise Correlation Heatmaps")
        for district in selected_district:
            st.markdown(f"**{district}**")
            st.plotly_chart(create_correlation_heatmap(corr_df, district=district), use_container_width=True)
        st.subheader("Correlation Data Table")
        st.dataframe(corr_df)

    # District Comparison
    with tabs[2]:
        st.subheader("Grouped Bar Chart: Average Correlations by District")
        st.plotly_chart(create_grouped_bar_chart(corr_df), use_container_width=True)
        st.subheader("Box Plots: Sales Distribution")
        st.plotly_chart(create_box_plots(df_filtered, by='district'), use_container_width=True)
        st.plotly_chart(create_box_plots(df_filtered, by='fiscal_year'), use_container_width=True)

    # Time Series Analysis
    with tabs[3]:
        st.subheader("Monthly Sales Trends by District")
        for district in selected_district:
            st.markdown(f"**{district}**")
            st.plotly_chart(create_time_series(df_filtered, district=district), use_container_width=True)
        st.subheader("Faceted Scatter Plots")
        for var in selected_vars:
            st.plotly_chart(create_faceted_scatter(df_filtered, variable=var), use_container_width=True)

    # Statistical Summary
    with tabs[4]:
        st.subheader("Key Insights & Interpretation")
        st.markdown(generate_insights(corr_df))
        st.markdown("---")
        st.write("**Correlation Strength Legend:**")
        st.write("- Weak: |r| < 0.3\n- Moderate: 0.3 ≤ |r| < 0.7\n- Strong: |r| ≥ 0.7")
        st.write("**Significance:** p < 0.05")

    # Export Results
    with tabs[5]:
        st.subheader("Download Correlation Results")
        st.markdown(export_results(corr_df), unsafe_allow_html=True)
        st.subheader("Export Plots")
        st.info("Right-click on any plot to save as PNG.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed for comprehensive district-wise sales-environment analysis.")

if __name__ == "__main__":
    main() 