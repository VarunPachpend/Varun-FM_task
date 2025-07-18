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
    pivot_col = 'fiscal_year' if district else 'district'
    # If duplicates exist, aggregate by mean
    if data.duplicated(subset=['variable', pivot_col]).any():
        dupes = data[data.duplicated(subset=['variable', pivot_col], keep=False)]
        st.warning(f'Duplicate entries found for heatmap and aggregated by mean. Example rows: {dupes.head(3).to_dict(orient="records")}')
        data = data.groupby(['variable', pivot_col], as_index=False)['correlation'].mean()
    heatmap_data = data.pivot(index='variable', columns=pivot_col, values='correlation')
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
    tabs = st.tabs(["Data Overview", "Correlation Analysis", "District Comparison", "Time Series Analysis", "Statistical Summary", "Executive Summary", "Export Results"])

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
        st.markdown(generate_chart_conclusion(corr_df, "heatmap"))
        
        st.subheader("District-wise Correlation Heatmaps")
        for district in selected_district:
            st.markdown(f"**{district}**")
            st.plotly_chart(create_correlation_heatmap(corr_df, district=district), use_container_width=True)
            st.markdown(generate_chart_conclusion(corr_df, "heatmap", district=district))
        
        st.subheader("Correlation Data Table")
        st.dataframe(corr_df)

    # District Comparison
    with tabs[2]:
        st.subheader("Grouped Bar Chart: Average Correlations by District")
        st.plotly_chart(create_grouped_bar_chart(corr_df), use_container_width=True)
        st.markdown(generate_chart_conclusion(corr_df, "bar"))
        
        st.subheader("Box Plots: Sales Distribution")
        st.plotly_chart(create_box_plots(df_filtered, by='district'), use_container_width=True)
        st.markdown(generate_chart_conclusion(corr_df, "box"))
        
        st.plotly_chart(create_box_plots(df_filtered, by='fiscal_year'), use_container_width=True)
        st.markdown(generate_chart_conclusion(corr_df, "box"))

    # Time Series Analysis
    with tabs[3]:
        st.subheader("Monthly Sales Trends by District")
        for district in selected_district:
            st.markdown(f"**{district}**")
            st.plotly_chart(create_time_series(df_filtered, district=district), use_container_width=True)
            st.markdown(generate_chart_conclusion(corr_df, "time_series", district=district))
        
        st.subheader("Faceted Scatter Plots")
        for var in selected_vars:
            st.plotly_chart(create_faceted_scatter(df_filtered, variable=var), use_container_width=True)
            st.markdown(generate_chart_conclusion(corr_df, "scatter"))

    # Statistical Summary
    with tabs[4]:
        st.subheader("Key Insights & Interpretation")
        st.markdown(generate_insights(corr_df))
        st.markdown("---")
        st.write("**Correlation Strength Legend:**")
        st.write("- Weak: |r| < 0.3\n- Moderate: 0.3 ≤ |r| < 0.7\n- Strong: |r| ≥ 0.7")
        st.write("**Significance:** p < 0.05")

    # Executive Summary
    with tabs[5]:
        st.markdown(generate_executive_summary(df_filtered, corr_df))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Impactful Environmental Factors")
            impact_chart = create_impact_ranking_chart(corr_df)
            if impact_chart:
                st.plotly_chart(impact_chart, use_container_width=True)
        
        with col2:
            st.subheader("District Performance Ranking")
            perf_chart = create_district_performance_chart(df_filtered)
            if perf_chart:
                st.plotly_chart(perf_chart, use_container_width=True)
        
        st.subheader("Quick Action Items")
        st.markdown("""
        **Based on the analysis, consider these strategic actions:**
        
        1. **Focus on the most impactful environmental factor** identified above
        2. **Develop targeted strategies** for underperforming districts
        3. **Monitor seasonal patterns** and adjust inventory accordingly
        4. **Investigate strong correlations** for business optimization opportunities
        5. **Set up regular monitoring** of key environmental indicators
        """)

    # Export Results
    with tabs[6]:
        st.subheader("Download Correlation Results")
        st.markdown(export_results(corr_df), unsafe_allow_html=True)
        st.subheader("Export Plots")
        st.info("Right-click on any plot to save as PNG.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed for comprehensive district-wise sales-environment analysis.")

# -----------------------------
# 12. Generate Chart Conclusions
# -----------------------------
def generate_chart_conclusion(corr_df, chart_type="heatmap", district=None, fiscal_year=None):
    """Generate conclusion statements for different chart types."""
    if corr_df.empty:
        return "No correlation data available for the selected filters."
    
    data = corr_df.copy()
    if district:
        data = data[data['district'] == district]
    if fiscal_year:
        data = data[data['fiscal_year'] == fiscal_year]
    
    if chart_type == "heatmap":
        # Overall correlation insights
        strong_pos = data[(data['correlation'] >= 0.7) & (data['p_value'] < 0.05)]
        strong_neg = data[(data['correlation'] <= -0.7) & (data['p_value'] < 0.05)]
        moderate = data[(data['correlation'].abs() >= 0.3) & (data['correlation'].abs() < 0.7) & (data['p_value'] < 0.05)]
        
        conclusion = f"**Correlation Analysis Summary:**\n\n"
        
        if not strong_pos.empty:
            top_pos = strong_pos.loc[strong_pos['correlation'].idxmax()]
            conclusion += f"• **Strongest Positive Correlation:** {top_pos['variable']} (r={top_pos['correlation']:.3f}, p={top_pos['p_value']:.3f})\n"
        
        if not strong_neg.empty:
            top_neg = strong_neg.loc[strong_neg['correlation'].idxmin()]
            conclusion += f"• **Strongest Negative Correlation:** {top_neg['variable']} (r={top_neg['correlation']:.3f}, p={top_neg['p_value']:.3f})\n"
        
        if not moderate.empty:
            conclusion += f"• **Moderate Correlations:** {len(moderate)} variables show moderate relationships\n"
        
        sig_count = len(data[data['p_value'] < 0.05])
        total_count = len(data)
        conclusion += f"• **Statistical Significance:** {sig_count}/{total_count} correlations are statistically significant (p<0.05)\n"
        
        if district:
            conclusion += f"\n**District-specific Insights for {district}:**\n"
            district_data = data[data['district'] == district]
            if not district_data.empty:
                top_var = district_data.loc[district_data['correlation'].abs().idxmax()]
                conclusion += f"• Most influential factor: {top_var['variable']} (r={top_var['correlation']:.3f}, {top_var['strength']} correlation)"
        
        return conclusion
    
    elif chart_type == "scatter":
        if district and fiscal_year:
            return f"**Scatter Plot Interpretation for {district}, {fiscal_year}:**\n\n• Each point represents a monthly observation\n• Trend line shows the overall relationship direction\n• Color coding indicates seasonal patterns\n• R² value shows the strength of the linear relationship"
        else:
            return "**Scatter Plot Interpretation:**\n\n• Each point represents a monthly observation\n• Trend line shows the overall relationship direction\n• Color coding indicates seasonal patterns"
    
    elif chart_type == "box":
        return "**Box Plot Interpretation:**\n\n• Box shows the interquartile range (IQR)\n• Median line indicates central tendency\n• Whiskers show data spread\n• Outliers indicate unusual sales patterns\n• Compare distributions across groups"
    
    elif chart_type == "time_series":
        if district:
            return f"**Time Series Analysis for {district}:**\n\n• Line shows monthly sales trends\n• Different colors represent fiscal years\n• Seasonal patterns and trends are visible\n• Peaks and troughs indicate business cycles"
        else:
            return "**Time Series Analysis:**\n\n• Line shows monthly sales trends\n• Different colors represent fiscal years\n• Seasonal patterns and trends are visible\n• Peaks and troughs indicate business cycles"
    
    elif chart_type == "bar":
        return "**Bar Chart Interpretation:**\n\n• Bars show average correlation strength by district\n• Positive values indicate positive relationships\n• Negative values indicate inverse relationships\n• Longer bars indicate stronger correlations\n• Compare relative importance across districts"
    
    else:
        return "**Chart Interpretation:**\n\n• Analyze patterns and relationships in the data\n• Look for trends, outliers, and significant correlations\n• Consider business implications of findings"

# -----------------------------
# 13. Executive Summary Functions
# -----------------------------
def generate_executive_summary(df, corr_df):
    """Generate comprehensive executive summary with key insights."""
    if df.empty or corr_df.empty:
        return "No data available for executive summary."
    
    # Data overview
    total_districts = df['district'].nunique()
    total_years = df['fiscal_year'].nunique()
    total_months = df['month'].nunique()
    avg_sales = df['sales_volume'].mean()
    total_obs = len(df)
    
    # Top correlations across all data
    all_corr = corr_df.copy()
    significant_corr = all_corr[all_corr['p_value'] < 0.05].copy()
    
    # Most impactful variables (highest absolute correlation)
    most_impactful = all_corr.loc[all_corr['correlation'].abs().idxmax()]
    
    # Top 5 positive and negative correlations
    top_positive = all_corr[all_corr['correlation'] > 0].nlargest(5, 'correlation')
    top_negative = all_corr[all_corr['correlation'] < 0].nsmallest(5, 'correlation')
    
    # District performance analysis
    district_sales = df.groupby('district')['sales_volume'].agg(['mean', 'std', 'count']).round(2)
    best_district = district_sales.loc[district_sales['mean'].idxmax()]
    worst_district = district_sales.loc[district_sales['mean'].idxmin()]
    
    # Year-over-year trends
    yearly_sales = df.groupby('fiscal_year')['sales_volume'].mean()
    sales_growth = ((yearly_sales.iloc[-1] - yearly_sales.iloc[0]) / yearly_sales.iloc[0] * 100).round(2)
    
    summary = f"""
## 📊 Executive Summary Dashboard

### 🎯 **Data Overview**
- **Total Observations:** {total_obs:,}
- **Districts Analyzed:** {total_districts}
- **Fiscal Years:** {total_years}
- **Time Period:** {total_months} months
- **Average Sales Volume:** {avg_sales:,.2f}

### 🔥 **Most Impactful Environmental Factor**
**{most_impactful['variable']}** shows the strongest relationship with sales:
- **Correlation:** {most_impactful['correlation']:.3f} ({most_impactful['strength']})
- **Statistical Significance:** p = {most_impactful['p_value']:.3f}
- **District:** {most_impactful['district']}
- **Year:** {most_impactful['fiscal_year']}

### 📈 **Top 5 Positive Correlations**
"""
    
    for idx, row in top_positive.iterrows():
        summary += f"- **{row['variable']}** (r={row['correlation']:.3f}, {row['strength']}) - {row['district']}, {row['fiscal_year']}\n"
    
    summary += f"\n### 📉 **Top 5 Negative Correlations**\n"
    
    for idx, row in top_negative.iterrows():
        summary += f"- **{row['variable']}** (r={row['correlation']:.3f}, {row['strength']}) - {row['district']}, {row['fiscal_year']}\n"
    
    summary += f"""
### 🏆 **District Performance Analysis**
- **Best Performing District:** {district_sales['mean'].idxmax()} (Avg: {best_district['mean']:,.2f})
- **Lowest Performing District:** {district_sales['mean'].idxmin()} (Avg: {worst_district['mean']:,.2f})
- **Sales Growth Trend:** {sales_growth}% over the analyzed period

### 📊 **Statistical Significance**
- **Significant Correlations:** {len(significant_corr)} out of {len(all_corr)} ({len(significant_corr)/len(all_corr)*100:.1f}%)
- **Strong Correlations (|r| ≥ 0.7):** {len(all_corr[all_corr['correlation'].abs() >= 0.7])}
- **Moderate Correlations (0.3 ≤ |r| < 0.7):** {len(all_corr[(all_corr['correlation'].abs() >= 0.3) & (all_corr['correlation'].abs() < 0.7)])}

### 💡 **Key Business Insights**
"""
    
    # Generate business insights
    if most_impactful['correlation'] > 0.5:
        summary += f"• **{most_impactful['variable']}** has a strong positive impact on sales - consider leveraging this relationship\n"
    elif most_impactful['correlation'] < -0.5:
        summary += f"• **{most_impactful['variable']}** has a strong negative impact on sales - consider mitigation strategies\n"
    
    if sales_growth > 0:
        summary += f"• Overall sales show a positive growth trend of {sales_growth}%\n"
    else:
        summary += f"• Sales show a declining trend of {abs(sales_growth)}% - investigate underlying factors\n"
    
    summary += f"• {len(significant_corr)} environmental factors significantly influence sales performance\n"
    summary += f"• District performance varies significantly - consider targeted strategies\n"
    
    return summary

def create_impact_ranking_chart(corr_df):
    """Create a chart showing the most impactful variables."""
    if corr_df.empty:
        return None
    
    # Get average absolute correlation by variable
    impact_ranking = corr_df.groupby('variable')['correlation'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    
    fig = px.bar(x=impact_ranking.values, y=impact_ranking.index, orientation='h',
                 title="Environmental Factors by Impact on Sales (Average Absolute Correlation)",
                 labels={'x': 'Average |Correlation|', 'y': 'Environmental Factor'})
    fig.update_layout(showlegend=False)
    return fig

def create_district_performance_chart(df):
    """Create a chart showing district performance."""
    if df.empty:
        return None
    
    district_perf = df.groupby('district')['sales_volume'].mean().sort_values(ascending=False)
    
    fig = px.bar(x=district_perf.index, y=district_perf.values,
                 title="Average Sales Volume by District",
                 labels={'x': 'District', 'y': 'Average Sales Volume'})
    fig.update_layout(showlegend=False)
    return fig

if __name__ == "__main__":
    main() 