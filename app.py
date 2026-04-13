# Core Packages
import base64
import codecs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="🧠",
    layout="wide",
)

# Add smooth animations and attractive light colors
st.markdown("""
<style>
    /* Color Palette - Soft Light Colors */
    :root {
        --primary: #5DA9E8;
        --secondary: #FFB347;
        --success: #66BB6A;
        --warning: #FFA500;
        --info: #AB47BC;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    @keyframes slideIn {
        from {opacity: 0; transform: translateX(-20px);}
        to {opacity: 1; transform: translateX(0);}
    }
    
    @keyframes slideInRight {
        from {opacity: 0; transform: translateX(20px);}
        to {opacity: 1; transform: translateX(0);}
    }
    
    @keyframes bounce {
        0%, 100% {transform: translateY(0);}
        50% {transform: translateY(-5px);}
    }
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #F5F7FF 0%, #FFF5F7 100%);
    }
    
    /* Metrics with soft light colors */
    [data-testid="metric-container"] {
        animation: fadeIn 0.6s ease-in-out;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid;
        color: #ffffff;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(93, 169, 232, 0.15);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:nth-child(1) {
        background: linear-gradient(135deg, #5DA9E8 0%, #4A8FC7 100%);
        border-left-color: #3A6FA7;
    }
    
    [data-testid="metric-container"]:nth-child(2) {
        background: linear-gradient(135deg, #FFB347 0%, #FFA500 100%);
        border-left-color: #FF8C00;
    }
    
    [data-testid="metric-container"]:nth-child(3) {
        background: linear-gradient(135deg, #66BB6A 0%, #4CAF50 100%);
        border-left-color: #388E3C;
    }
    
    [data-testid="metric-container"]:nth-child(4) {
        background: linear-gradient(135deg, #AB47BC 0%, #8E24AA 100%);
        border-left-color: #6A1B9A;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(93, 169, 232, 0.25);
    }
    }
    
    /* Buttons */
    .stButton button {
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        background: linear-gradient(135deg, #5DA9E8 0%, #4A8FC7 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-size: 14px !important;
        box-shadow: 0 4px 15px rgba(93, 169, 232, 0.25) !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #4A8FC7 0%, #3A6FA7 100%) !important;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(93, 169, 232, 0.4) !important;
    }
    
    .stButton button:active {
        transform: translateY(-1px);
    }
    
    /* Expanders */
    .stExpander {
        animation: slideIn 0.5s ease-in-out;
        border-left: 4px solid #5DA9E8 !important;
        border-radius: 10px !important;
        background: #FAFBFF !important;
    }
    
    .streamlit-expanderHeader {
        color: #5DA9E8 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }
    
    .streamlit-expanderHeader:hover {
        color: #4A8FC7 !important;
    }
    
    /* Headers and Text */
    h1 {
        color: #5DA9E8 !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 4px rgba(93, 169, 232, 0.1);
        animation: fadeIn 0.8s ease-in-out;
    }
    
    h2 {
        color: #5DA9E8 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #FFB347;
        padding-bottom: 10px;
        animation: slideIn 0.6s ease-in-out;
    }
    
    h3 {
        color: #4A8FC7 !important;
        font-weight: 700 !important;
        animation: slideIn 0.6s ease-in-out;
    }
    
    h4 {
        color: #5DA9E8 !important;
        font-weight: 600 !important;
    }
    
    /* Normal text - dark gray for good contrast */
    p, label, span {
        color: #424242 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        animation: fadeIn 0.6s ease-in-out;
        border-radius: 10px !important;
        border: 2px solid #E3F2FD !important;
        box-shadow: 0 2px 10px rgba(93, 169, 232, 0.12) !important;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%) !important;
        border-left: 5px solid #66BB6A !important;
        color: #2E7D32 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    .stSuccess p {
        color: #2E7D32 !important;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%) !important;
        border-left: 5px solid #5DA9E8 !important;
        color: #1565C0 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    .stInfo p {
        color: #1565C0 !important;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFF3E0 100%) !important;
        border-left: 5px solid #FFB347 !important;
        color: #E65100 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    .stWarning p {
        color: #E65100 !important;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(135deg, #FCE4EC 0%, #F3E5F5 100%) !important;
        border-left: 5px solid #FF6B9D !important;
        color: #C2185B !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    .stError p {
        color: #C2185B !important;
    }
    
    /* Selectbox and inputs */
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
        animation: slideIn 0.5s ease-in-out;
    }
    
    .stCheckbox {
        animation: slideInRight 0.5s ease-in-out;
    }
    
    /* Selectbox styling */
    .stSelectbox label, .stMultiSelect label {
        color: #424242 !important;
        font-weight: 600 !important;
    }
    
    /* Horizontal line */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #5DA9E8 0%, #FFB347 50%, #66BB6A 100%);
        margin: 20px 0;
        border-radius: 2px;
    }
    
    /* Markdown content */
    .stMarkdown {
        animation: fadeIn 0.7s ease-in-out;
    }
    
    /* Metric label and value styling */
    [data-testid="metricDeltaContainer"], [data-testid="metricDeltaValue"] {
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F0F6FF 0%, #FFF5F7 100%);
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #424242 !important;
    }
</style>
""", unsafe_allow_html=True)


def render_html_report(path, height=700):
    with codecs.open(path, "r", encoding="utf-8") as file:
        html = file.read()
    components.html(html, height=height, scrolling=True)


def download_csv(df, filename="processed_data.csv"):
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f"<a href=\"data:file/csv;base64,{b64}\" download=\"{filename}\">Download processed CSV</a>"
    st.markdown(href, unsafe_allow_html=True)


def load_dataset(uploaded_file):
    return pd.read_csv(uploaded_file)


def dataset_metrics(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Numeric cols": len(df.select_dtypes(include=[np.number]).columns),
        "Missing cells": int(df.isna().sum().sum()),
    }


def render_basic_eda(df):
    st.subheader("🔍 Exploratory Data Analysis")
    st.write("Use the controls below to inspect the uploaded dataset.")

    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    left, right = st.columns([2, 1])
    with left:
        if st.checkbox("📏 Show Shape"):
            st.write(df.shape)

        if st.checkbox("📋 Show Columns"):
            st.write(all_columns)

        if st.checkbox("🎯 Show Data Types"):
            st.write(df.dtypes)

        if st.checkbox("❌ Null value count"):
            st.write(df.isnull().sum())

        if st.checkbox("📊 Summary"):
            st.write(df.describe(include="all").T)

        if st.checkbox("✏️ Show Selected Columns"):
            selected_columns = st.multiselect("Select Columns", all_columns)
            if selected_columns:
                st.dataframe(df[selected_columns])
            else:
                st.info("Pick one or more columns.")

        if st.checkbox("🏷️ Value Counts For categorical values"):
            if categorical_columns:
                selected_column = st.selectbox("Select a Column", categorical_columns, key="value_counts_col")
                st.write(df[selected_column].value_counts())
            else:
                st.warning("No categorical columns available.")

    with right:
        st.markdown("<h4 style='color: #667eea; text-align: center;'>📊 Top-level overview</h4>", unsafe_allow_html=True)
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.metric("Numeric", len(numeric_columns))
        st.metric("Categorical", len(categorical_columns))

        if categorical_columns:
            st.markdown("<h4 style='color: #764ba2; text-align: center;'>🏷️ Categories</h4>", unsafe_allow_html=True)
            st.write(categorical_columns)

    st.markdown("---")
    if st.checkbox("🔥 Show correlation heatmap"):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="vlag", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("At least two numeric columns are required for a correlation matrix.")

    if st.checkbox("Show top-level statistics"):
        st.write(df.describe(include="all").T)


def render_sweetviz(df):
    st.subheader("✨ Sweetviz Report")
    st.write("Generate an interactive Sweetviz report for the uploaded dataset.")
    if st.button("🚀 Run Sweetviz analysis"):
        report = sv.analyze(df)
        report.show_html("sweetviz_report.html")
        render_html_report("sweetviz_report.html")


def render_plots(df):
    st.subheader("📊 Automatic Plot Gallery")
    st.write("Important plots are generated automatically for the dataset.")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    selected_numeric = st.multiselect("📈 Numeric columns", numeric_columns, default=numeric_columns[:3])
    selected_categorical = st.multiselect("🏷️ Categorical columns", categorical_columns, default=categorical_columns[:1])

    if not selected_numeric and not selected_categorical:
        st.warning("Select at least one numeric or categorical column to render the plot gallery.")
        return

    if selected_numeric:
        with st.expander("📊 Histograms"):
            for col in selected_numeric:
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#667eea')
                ax.set_title(f"📊 Histogram: {col}", fontsize=14, fontweight='bold', color='#667eea')
                st.pyplot(fig)

        with st.expander("📦 Boxplots"):
            fig, ax = plt.subplots(figsize=(8, max(4, len(selected_numeric) * 1.5)))
            sns.boxplot(data=df[selected_numeric], orient="h", ax=ax, palette='Set2')
            ax.set_title("📦 Boxplot for numeric columns", fontsize=14, fontweight='bold', color='#667eea')
            st.pyplot(fig)

        if len(selected_numeric) >= 2:
            with st.expander("🔥 Correlation heatmap"):
                corr_df = df[selected_numeric].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("🔥 Correlation matrix", fontsize=14, fontweight='bold', color='#667eea')
                st.pyplot(fig)

            with st.expander("🎯 Scatter matrix"):
                pair_cols = selected_numeric[:4]
                fig = pd.plotting.scatter_matrix(df[pair_cols].dropna(), figsize=(10, 10), diagonal="kde", color='#667eea')
                st.pyplot(fig[0, 0].figure)

    if selected_categorical:
        with st.expander("📈 Categorical counts"):
            for col in selected_categorical:
                fig, ax = plt.subplots(figsize=(8, 4))
                df[col].value_counts().nlargest(10).plot(kind="bar", ax=ax, color='#f5576c')
                ax.set_title(f"📈 Value counts: {col}", fontsize=14, fontweight='bold', color='#667eea')
                ax.set_ylabel("Count", fontweight='bold')
                ax.set_xlabel(col, fontweight='bold')
                st.pyplot(fig)

    if selected_numeric:
        with st.expander("📉 Line chart for numeric series"):
            fig, ax = plt.subplots(figsize=(10, 4))
            df[selected_numeric].dropna().plot(ax=ax, color=['#667eea', '#f5576c', '#43e97b', '#fa709a'][:len(selected_numeric)], linewidth=2.5)
            ax.set_title("📉 Line chart for selected numeric columns", fontsize=14, fontweight='bold', color='#667eea')
            ax.legend(loc='best', framealpha=0.9)
            st.pyplot(fig)


def render_preprocess(df):
    st.subheader("✨ Preprocessed Data")
    st.write("This section automatically presents the cleaned dataset without extra selection options.")

    clean_df = df.dropna().reset_index(drop=True)
    removed_rows = df.shape[0] - clean_df.shape[0]

    if removed_rows > 0:
        st.success(f"✅ Removed {removed_rows} rows with missing values.")
    else:
        st.success("✅ No missing values found; data is already clean.")

    st.markdown("<h4 style='color: #667eea;'>📋 Processed dataset preview</h4>", unsafe_allow_html=True)
    st.dataframe(clean_df.head(20), use_container_width=True)
    st.markdown("<h4 style='color: #764ba2;'>📊 Processed dataset details</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", clean_df.shape[0])
    with col2:
        st.metric("Columns", clean_df.shape[1])

    download_csv(clean_df, filename="preprocessed_data.csv")


def render_missing_values(df):
    st.subheader("📊 Handle Missing Values")
    st.write("Analyze and manage missing data in your dataset with multiple strategies.")
    
    # Missing value overview
    col1, col2, col3, col4 = st.columns(4)
    missing_vals = df.isnull().sum().sum()
    missing_percent = (missing_vals / (df.shape[0] * df.shape[1])) * 100
    
    with col1:
        st.metric("🔴 Total Missing Cells", missing_vals)
    with col2:
        st.metric("📉 Missing %", f"{missing_percent:.2f}%")
    with col3:
        st.metric("✅ Complete Rows", (df.dropna().shape[0]))
    with col4:
        st.metric("⚠️ Affected Columns", df.columns[df.isnull().any()].shape[0])
    
    st.markdown("---")
    
    # Missing value visualization
    st.markdown("<h4 style='color: #667eea;'>🔍 Missing Value Pattern</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            missing_data[missing_data > 0].plot(kind='barh', ax=ax, color='#f5576c', edgecolor='#667eea', linewidth=2)
            ax.set_xlabel('Number of Missing Values', fontweight='bold')
            ax.set_title('Missing Values by Column', fontsize=13, fontweight='bold', color='#667eea')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
        else:
            st.success("✅ No missing values detected!")
    
    with col1:
        st.markdown("<h4 style='color: #764ba2;'>📊 Missing Value Percentage</h4>", unsafe_allow_html=True)
        missing_percent_data = (df.isnull().sum() / len(df)) * 100
        if missing_percent_data.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            missing_percent_data[missing_percent_data > 0].plot(kind='barh', ax=ax, color='#4facfe', edgecolor='#667eea', linewidth=2)
            ax.set_xlabel('Missing Percentage (%)', fontweight='bold')
            ax.set_title('Missing Percentage by Column', fontsize=13, fontweight='bold', color='#667eea')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
    
    st.markdown("---")
    
    # Handling strategies
    st.markdown("<h4 style='color: #667eea;'>🛠️ Choose Handling Strategy</h4>", unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    if not cols_with_missing:
        st.info("No columns with missing values found!")
        return
    
    strategy = st.selectbox("🎯 Select a strategy", [
        "Drop rows with missing values",
        "Fill with mean (numeric columns)",
        "Fill with median (numeric columns)",
        "Fill with mode (most frequent)",
        "Forward fill",
        "Backward fill",
        "Linear interpolation (numeric only)",
        "Custom value fill"
    ])
    
    # Apply selected strategy
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h4 style='color: #764ba2;'>👁️ Preview of Processed Data</h4>", unsafe_allow_html=True)
        
        if strategy == "Drop rows with missing values":
            processed_df = df.dropna().reset_index(drop=True)
            st.info(f"📊 Removed {df.shape[0] - processed_df.shape[0]} rows with missing values")
            
        elif strategy == "Fill with mean (numeric columns)":
            processed_df = df.copy()
            for col in numeric_cols:
                if processed_df[col].isnull().sum() > 0:
                    processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            st.info("📈 Filled numeric columns with mean values")
            
        elif strategy == "Fill with median (numeric columns)":
            processed_df = df.copy()
            for col in numeric_cols:
                if processed_df[col].isnull().sum() > 0:
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
            st.info("📊 Filled numeric columns with median values")
            
        elif strategy == "Fill with mode (most frequent)":
            processed_df = df.copy()
            for col in df.columns:
                if processed_df[col].isnull().sum() > 0:
                    mode_val = processed_df[col].mode()
                    if len(mode_val) > 0:
                        processed_df[col].fillna(mode_val[0], inplace=True)
            st.info("🏷️ Filled columns with mode (most frequent value)")
            
        elif strategy == "Forward fill":
            processed_df = df.fillna(method='ffill').fillna(method='bfill')
            st.info("➡️ Applied forward fill and then backward fill")
            
        elif strategy == "Backward fill":
            processed_df = df.fillna(method='bfill').fillna(method='ffill')
            st.info("⬅️ Applied backward fill and then forward fill")
            
        elif strategy == "Linear interpolation (numeric only)":
            processed_df = df.copy()
            for col in numeric_cols:
                processed_df[col] = processed_df[col].interpolate(method='linear')
            st.info("📈 Applied linear interpolation to numeric columns")
            
        elif strategy == "Custom value fill":
            fill_value = st.text_input("✏️ Enter value to fill missing data with:", "0")
            try:
                processed_df = df.fillna(fill_value)
                st.info(f"✅ Filled missing values with: {fill_value}")
            except:
                st.error("❌ Invalid fill value")
                return
        
        st.dataframe(processed_df.head(15), use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='color: #764ba2; text-align: center;'>📈 Statistics</h4>", unsafe_allow_html=True)
        remaining_missing = processed_df.isnull().sum().sum()
        st.metric("Remaining Missing Values", remaining_missing)
        st.metric("Rows After Processing", processed_df.shape[0])
        st.metric("Columns", processed_df.shape[1])
    
    st.markdown("---")
    
    # Download processed data
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color: #667eea;'>💾 Export Processed Data</h4>", unsafe_allow_html=True)
        download_csv(processed_df, filename="handled_missing_values.csv")
    
    with col2:
        if st.button("📋 Show Processing Summary"):
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea15, #764ba215); padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;'>
            <b>Processing Summary:</b><br>
            📊 Original rows: <b>{df.shape[0]}</b><br>
            ✅ Final rows: <b>{processed_df.shape[0]}</b><br>
            ❌ Rows removed: <b>{df.shape[0] - processed_df.shape[0]}</b><br>
            🔴 Original missing values: <b>{df.isnull().sum().sum()}</b><br>
            ✨ Final missing values: <b>{processed_df.isnull().sum().sum()}</b>
            </div>
            """, unsafe_allow_html=True)


def render_models(df):
    st.subheader("🤖 Machine Learning Models")
    st.write("Train and compare multiple ML models on your dataset.")
    
    # Prepare data for modeling
    st.markdown("<h4 style='color: #667eea;'>📊 Model Configuration</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Numeric Columns (Features):**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("⚠️ No numeric columns found. Models require numeric features.")
            return
        selected_features = st.multiselect("Select feature columns", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
    
    with col2:
        st.write("**Target Column:**")
        all_cols = df.columns.tolist()
        target_col = st.selectbox("Select target column for classification", all_cols)
    
    if not selected_features or target_col not in all_cols:
        st.error("❌ Please select features and target column")
        return
    
    st.markdown("---")
    
    # Prepare data
    X = df[selected_features].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) == 0:
        st.error("❌ No valid data after removing missing values")
        return
    
    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        target_classes = le.classes_
    else:
        target_classes = np.unique(y)
    
    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Samples", len(X))
    with col2:
        st.metric("🔢 Features", len(selected_features))
    with col3:
        st.metric("🏷️ Classes", len(target_classes))
    with col4:
        st.metric("✅ Valid Data %", f"{(len(X)/len(df))*100:.1f}%")
    
    st.markdown("---")
    
    # Train test split
    test_size = st.slider("Test Set Size %", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.markdown("<h4 style='color: #667eea;'>🚀 Training Models...</h4>", unsafe_allow_html=True)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(random_state=42),
        'Naive Bayes': GaussianNB(),
    }
    
    # Train models and store results
    results = []
    predictions = {}
    trained_models = {}
    
    progress_bar = st.progress(0)
    for idx, (name, model) in enumerate(models.items()):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'Model': name,
                'Accuracy': f"{accuracy*100:.2f}%",
                'Precision': f"{precision*100:.2f}%",
                'Recall': f"{recall*100:.2f}%",
                'F1-Score': f"{f1*100:.2f}%",
                'Accuracy (float)': accuracy,
            })
            
            predictions[name] = y_pred
            trained_models[name] = model
        except Exception as e:
            st.warning(f"⚠️ {name} failed: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(models))
    
    results_df = pd.DataFrame(results)
    st.success("✅ Model training complete!")
    
    st.markdown("---")
    st.markdown("<h4 style='color: #764ba2;'>📊 Model Performance Comparison</h4>", unsafe_allow_html=True)
    
    # Show results table
    st.dataframe(results_df.drop('Accuracy (float)', axis=1), use_container_width=True)
    
    st.markdown("---")
    
    # Visualize accuracy comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h5 style='color: #667eea;'>📈 Accuracy Comparison</h5>", unsafe_allow_html=True)
        accuracy_data = results_df.sort_values('Accuracy (float)', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#FAFBFF')
        ax.set_facecolor('#F5F7FA')
        
        colors = ['#66BB6A' if x == accuracy_data['Accuracy (float)'].max() else '#5DA9E8' 
                 for x in accuracy_data['Accuracy (float)']]
        
        bars = ax.barh(accuracy_data['Model'], accuracy_data['Accuracy (float)'], color=colors, edgecolor='white', linewidth=2)
        ax.set_xlabel('Accuracy Score', fontsize=11, fontweight='bold', color='#424242')
        ax.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold', color='#667eea')
        ax.set_xlim([0, 1.05])
        ax.tick_params(axis='both', labelcolor='#424242')
        ax.grid(axis='x', alpha=0.3, color='#E8EAEF', linestyle='--')
        
        # Add percentage labels
        for i, (idx, row) in enumerate(accuracy_data.iterrows()):
            ax.text(row['Accuracy (float)'] + 0.02, i, f"{row['Accuracy (float)']*100:.1f}%", 
                   va='center', fontweight='bold', color='#424242', fontsize=10)
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("<h5 style='color: #764ba2;'>🏆 Best Model</h5>", unsafe_allow_html=True)
        best_idx = results_df['Accuracy (float)'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        best_accuracy = results_df.loc[best_idx, 'Accuracy (float)']
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #66BB6A 0%, #4CAF50 100%); 
                    padding: 25px; border-radius: 15px; text-align: center; color: white;'>
            <h3 style='margin: 0; font-size: 24px;'>{best_model_name}</h3>
            <p style='margin: 10px 0 0 0; font-size: 18px; font-weight: 600;'>{best_accuracy*100:.2f}% Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h5 style='color: #667eea; margin-top: 20px;'>📊 Performance Metrics</h5>", unsafe_allow_html=True)
        best_row = results_df.loc[best_idx]
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Precision", best_row['Precision'])
            st.metric("Recall", best_row['Recall'])
        with col_b:
            st.metric("F1-Score", best_row['F1-Score'])
            st.metric("Test Samples", len(X_test))
    
    st.markdown("---")
    
    # Confusion Matrix for best model
    if best_model_name in trained_models:
        st.markdown("<h4 style='color: #667eea;'>🔍 Best Model - Confusion Matrix</h4>", unsafe_allow_html=True)
        
        best_predictions = predictions[best_model_name]
        cm = confusion_matrix(y_test, best_predictions)
        
        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#FAFBFF')
            ax.set_facecolor('#F5F7FA')
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Pastel1', cbar=True, 
                       xticklabels=target_classes, yticklabels=target_classes,
                       ax=ax, cbar_kws={'label': 'Count'}, linewidths=2.5, linecolor='white',
                       annot_kws={'size': 12, 'weight': 'bold', 'color': '#424242'})
            
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', color='#424242')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold', color='#424242')
            ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold', color='#667eea')
            ax.tick_params(axis='both', labelcolor='#424242')
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("<h5 style='color: #764ba2;'>📈 Classification Report</h5>", unsafe_allow_html=True)
            report = classification_report(y_test, best_predictions, target_names=target_classes, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.iloc[:-3], use_container_width=True)


def main():
    st.sidebar.markdown("""
    <h2 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               background-clip: text; text-align: center; padding: 10px;'>
    🚀 Activity selector
    </h2>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("""
    <p style='color: #667eea; text-align: center; font-weight: 600;'>
    Upload a dataset once and then pick the analysis panel.
    </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV file", type=["csv"])
    activity = st.sidebar.radio(
        "📊 Choose view",
        ["EDA(basic)", "Sweetviz", "Plots", "Handle Missing Values", "Preprocess Data", "🤖 ML Models"],
    )

    if uploaded_file is None:
        st.markdown("""
        <h1 style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   background-clip: text;'>🧠 EDA Dashboard</h1>
        """, unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align: center; color: #764ba2;'>Use the sidebar to upload a CSV file and explore the dataset using multiple analysis panels.</h4>",
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(245, 87, 108, 0.1)); 
                    padding: 30px; border-radius: 15px; border-left: 5px solid #667eea; text-align: center;'>
            <h3 style='color: #667eea; margin-top: 0;'>✨ Start Your Data Analysis Journey ✨</h3>
            <p style='color: #764ba2; font-size: 16px; font-weight: 600;'>
            📤 Start by uploading a CSV file in the left sidebar to begin exploring!
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = load_dataset(uploaded_file)

    st.markdown(f"""
    <h2 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               background-clip: text;'>
    📊 Dataset: <span style='color: #667eea;'><b>{uploaded_file.name}</b></span>
    </h2>
    """, unsafe_allow_html=True)
    
    if activity == "EDA(basic)":
        render_basic_eda(df)
    elif activity == "Sweetviz":
        render_sweetviz(df)
    elif activity == "Plots":
        render_plots(df)
    elif activity == "Handle Missing Values":
        render_missing_values(df)
    elif activity == "Preprocess Data":
        render_preprocess(df)
    elif activity == "🤖 ML Models":
        render_models(df)


if __name__ == "__main__":
    main()
