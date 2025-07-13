# District-wise Sales & Environmental Correlation Dashboard

A comprehensive Streamlit dashboard for analyzing district-wise sales volume correlations with environmental factors across fiscal years. Designed for interactive exploration, statistical insight, and business decision support.

---

## 🚀 Project Overview
This app enables users to:
- Explore how environmental factors (temperature, wind, rainfall, NDVI, etc.) relate to sales volume
- Analyze correlations by district and fiscal year
- Visualize trends, distributions, and relationships interactively
- Export results and gain actionable business insights

---

## 🧩 Features
- **Data Overview:** Inspect and validate your data
- **Correlation Analysis:** Pearson correlation matrices, p-values, and strength categories
- **District Comparison:** Grouped bar charts, box plots, and heatmaps
- **Time Series Analysis:** Monthly sales trends and faceted scatter plots
- **Statistical Summary:** Key insights, top influencers, and significance testing
- **Export Results:** Download correlation data and save plots
- **Sidebar Filters:** Select districts, fiscal year range, and variables
- **Interactive Visuals:** All major charts are interactive (Plotly)

---

## 📂 Data Requirements
- **File:** `Data.xlsx` (Excel format)
- **Columns:**
  - `district`, `month`, `fiscal_year`, `sales_volume`,
  - `temp_mean`, `wind_mean`, `gdd_cumulative`,
  - `temp_zscore`, `wind_zscore`, `gdd_zscore`,
  - `rainfall_avg`, `ndvi_mean`, `ndvi_zscore`
- Place `Data.xlsx` in the same directory as `app.py`.

---

## ⚙️ Setup & Installation
1. **Clone or download this repository.**
2. **Install required Python libraries:**
   ```bash
   pip install streamlit pandas numpy plotly seaborn matplotlib scikit-learn openpyxl
   ```
3. **Ensure your data file is named `Data.xlsx` and is in the project folder.**
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 🖥️ Usage
- Use the **sidebar** to filter by district, fiscal year, and environmental variables.
- Navigate between tabs for data overview, analysis, and exports.
- Hover, zoom, and select on interactive charts.
- Download correlation results as CSV from the Export tab.
- Right-click on any plot to save as PNG.

---

## 📊 Output & Interpretation
- **Correlation Matrix:** District-wise Pearson r and p-values
- **Strength Categories:** Weak/Moderate/Strong
- **Significance:** p < 0.05 highlighted
- **Top Influencers:** Most correlated factors per district
- **Business Insights:** Automated summary and recommendations

---

## 🛠️ Customization & Extensibility
- Modular code: Add new analyses or visualizations easily
- Supports large datasets with efficient caching
- Publication-ready plots and mobile-responsive layout

---

## ❓ Troubleshooting
- **Missing columns/data:** Ensure your Excel file matches the required structure
- **App not loading:** Check for errors in the terminal and required package installation
- **Plot issues:** Try refreshing the browser or restarting the app

---

## 📬 Contact & Support
For questions, suggestions, or support, please contact the project maintainer.

---

**Developed for comprehensive district-wise sales-environment analysis and business insight.** 