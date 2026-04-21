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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
import json
import os
from datetime import datetime

# ==================== FEATURE 1: DATA UPLOAD HISTORY ====================

def get_upload_history_file():
    """Get path to upload history JSON file"""
    return os.path.expanduser("~/.eda_dashboard_history.json")

def load_upload_history():
    """Load upload history from file"""
    history_file = get_upload_history_file()
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_upload_history(history):
    """Save upload history to file"""
    history_file = get_upload_history_file()
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save upload history: {e}")

def add_to_upload_history(filename, filepath):
    """Add file to upload history"""
    history = load_upload_history()
    
    # Check if file already exists in history
    for item in history:
        if item['filename'] == filename:
            # Update timestamp
            item['last_accessed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            item['access_count'] = item.get('access_count', 0) + 1
            save_upload_history(history)
            return
    
    # Add new entry
    history.insert(0, {
        'filename': filename,
        'filepath': filepath,
        'first_uploaded': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'last_accessed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'access_count': 1
    })
    
    # Keep only last 10 files
    history = history[:10]
    save_upload_history(history)

# ==================== FEATURE 2: SAVED PIPELINES ====================

def get_pipelines_file():
    """Get path to saved pipelines JSON file"""
    return os.path.expanduser("~/.eda_dashboard_pipelines.json")

def load_saved_pipelines():
    """Load saved pipelines from file"""
    pipelines_file = get_pipelines_file()
    if os.path.exists(pipelines_file):
        try:
            with open(pipelines_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_pipeline(pipeline_name, preprocessing_steps):
    """Save a preprocessing pipeline"""
    pipelines = load_saved_pipelines()
    pipelines[pipeline_name] = {
        'steps': preprocessing_steps,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'description': f"Pipeline with {len(preprocessing_steps)} steps"
    }
    
    pipelines_file = get_pipelines_file()
    try:
        with open(pipelines_file, 'w') as f:
            json.dump(pipelines, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Could not save pipeline: {e}")
        return False

def delete_pipeline(pipeline_name):
    """Delete a saved pipeline"""
    pipelines = load_saved_pipelines()
    if pipeline_name in pipelines:
        del pipelines[pipeline_name]
        pipelines_file = get_pipelines_file()
        try:
            with open(pipelines_file, 'w') as f:
                json.dump(pipelines, f, indent=2)
            return True
        except:
            return False
    return False

# ==================== FEATURE 3: UNDO/REDO FUNCTIONALITY ====================

def init_undo_redo_state():
    """Initialize undo/redo state in session"""
    if 'undo_stack' not in st.session_state:
        st.session_state.undo_stack = []
    if 'redo_stack' not in st.session_state:
        st.session_state.redo_stack = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None

def push_to_undo_stack(df, action_description):
    """Push dataframe state to undo stack"""
    if 'undo_stack' not in st.session_state:
        init_undo_redo_state()
    
    st.session_state.undo_stack.append({
        'df': df.copy(),
        'action': action_description,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Limit stack to 20 items
    if len(st.session_state.undo_stack) > 20:
        st.session_state.undo_stack.pop(0)
    
    # Clear redo stack on new action
    st.session_state.redo_stack = []

def undo_action():
    """Undo last action"""
    if 'undo_stack' not in st.session_state:
        init_undo_redo_state()
    
    if len(st.session_state.undo_stack) > 0:
        state = st.session_state.undo_stack.pop()
        st.session_state.redo_stack.append(state)
        return state
    return None

def redo_action():
    """Redo last undone action"""
    if 'redo_stack' not in st.session_state:
        init_undo_redo_state()
    
    if len(st.session_state.redo_stack) > 0:
        state = st.session_state.redo_stack.pop()
        st.session_state.undo_stack.append(state)
        return state
    return None

# ==================== FEATURE 4: TUTORIAL MODE ====================

def get_tutorial_config():
    """Get tutorial configuration"""
    return {
        'welcome': {
            'title': '👋 Welcome to EDA Dashboard!',
            'content': '''This is an interactive Exploratory Data Analysis (EDA) tool that helps you:
            - Load and explore CSV datasets
            - Visualize data distributions
            - Preprocess and clean data
            - Train machine learning models
            - Compare model performance
            
            **Tip:** Start by uploading a CSV file in the sidebar!'''
        },
        'eda_basic': {
            'title': '📊 Basic EDA Guide',
            'content': '''Here you can:
            - View dataset shape and columns
            - Check data types and missing values
            - See summary statistics
            - Explore correlation heatmap
            - View column value distributions
            
            **Pro Tip:** Use the checkboxes to toggle different analyses!'''
        },
        'preprocessing': {
            'title': '🔧 Data Preprocessing Guide',
            'content': '''Prepare your data by:
            - Handling missing values (drop, fill, etc.)
            - Detecting and treating outliers
            - Encoding categorical variables
            - Scaling/normalizing numeric features
            - Feature selection
            
            **Pro Tip:** Save your preprocessing steps as a pipeline for reuse!'''
        },
        'ml_models': {
            'title': '🤖 Machine Learning Guide',
            'content': '''Train and compare models:
            - Select target and feature columns
            - View data insights and recommendations
            - Train multiple models automatically
            - Compare performance metrics
            - Fine-tune hyperparameters
            - Export predictions
            
            **Pro Tip:** Check the data quality recommendations before training!'''
        }
    }

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="",
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

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)


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
    st.subheader("Exploratory Data Analysis")
    st.write("Use the controls below to inspect the uploaded dataset.")

    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    left, right = st.columns([2, 1])
    with left:
        if st.checkbox("Show Shape"):
            st.write(df.shape)

        if st.checkbox("Show Columns"):
            st.write(all_columns)

        if st.checkbox("Show Data Types"):
            st.write(df.dtypes)

        if st.checkbox("Null value count"):
            st.write(df.isnull().sum())

        if st.checkbox("Summary"):
            st.write(df.describe(include="all").T)

        if st.checkbox("Show Selected Columns"):
            selected_columns = st.multiselect("Select Columns", all_columns)
            if selected_columns:
                st.dataframe(df[selected_columns])
            else:
                st.info("Pick one or more columns.")

        if st.checkbox("Value Counts For categorical values"):
            if categorical_columns:
                selected_column = st.selectbox("Select a Column", categorical_columns, key="value_counts_col")
                st.write(df[selected_column].value_counts())
            else:
                st.warning("No categorical columns available.")

    with right:
        st.markdown("<h4 style='color: #667eea; text-align: center;'>Top-level overview</h4>", unsafe_allow_html=True)
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.metric("Numeric", len(numeric_columns))
        st.metric("Categorical", len(categorical_columns))

        if categorical_columns:
            st.markdown("<h4 style='color: #764ba2; text-align: center;'>Categories</h4>", unsafe_allow_html=True)
            st.write(categorical_columns)

    st.markdown("---")
    if st.checkbox("Show correlation heatmap"):
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
    st.subheader("Sweetviz Report")
    st.write("Generate an interactive Sweetviz report for the uploaded dataset.")
    if st.button("Run Sweetviz analysis"):
        report = sv.analyze(df)
        report.show_html("sweetviz_report.html")
        render_html_report("sweetviz_report.html")


def render_plots(df):
    st.subheader("Automatic Plot Gallery")
    st.write("Important plots are generated automatically for the dataset.")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    selected_numeric = st.multiselect("Numeric columns", numeric_columns, default=numeric_columns[:3])
    selected_categorical = st.multiselect("Categorical columns", categorical_columns, default=categorical_columns[:1])

    if not selected_numeric and not selected_categorical:
        st.warning("Select at least one numeric or categorical column to render the plot gallery.")
        return

    if selected_numeric:
        with st.expander("Histograms"):
            for col in selected_numeric:
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#667eea')
                ax.set_title(f"Histogram: {col}", fontsize=14, fontweight='bold', color='#667eea')
                st.pyplot(fig)

        with st.expander("Boxplots"):
            fig, ax = plt.subplots(figsize=(8, max(4, len(selected_numeric) * 1.5)))
            sns.boxplot(data=df[selected_numeric], orient="h", ax=ax, palette='Set2')
            ax.set_title("Boxplot for numeric columns", fontsize=14, fontweight='bold', color='#667eea')
            st.pyplot(fig)

        if len(selected_numeric) >= 2:
            with st.expander("Correlation heatmap"):
                corr_df = df[selected_numeric].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation matrix", fontsize=14, fontweight='bold', color='#667eea')
                st.pyplot(fig)

            with st.expander("Scatter matrix"):
                pair_cols = selected_numeric[:4]
                fig = pd.plotting.scatter_matrix(df[pair_cols].dropna(), figsize=(10, 10), diagonal="kde", color='#667eea')
                st.pyplot(fig[0, 0].figure)

    if selected_categorical:
        with st.expander("Categorical counts"):
            for col in selected_categorical:
                fig, ax = plt.subplots(figsize=(8, 4))
                df[col].value_counts().nlargest(10).plot(kind="bar", ax=ax, color='#f5576c')
                ax.set_title(f"Value counts: {col}", fontsize=14, fontweight='bold', color='#667eea')
                ax.set_ylabel("Count", fontweight='bold')
                ax.set_xlabel(col, fontweight='bold')
                st.pyplot(fig)

    if selected_numeric:
        with st.expander("Line chart for numeric series"):
            fig, ax = plt.subplots(figsize=(10, 4))
            df[selected_numeric].dropna().plot(ax=ax, color=['#667eea', '#f5576c', '#43e97b', '#fa709a'][:len(selected_numeric)], linewidth=2.5)
            ax.set_title("Line chart for selected numeric columns", fontsize=14, fontweight='bold', color='#667eea')
            ax.legend(loc='best', framealpha=0.9)
            st.pyplot(fig)


def render_preprocess(df):
    st.subheader("Complete Data Preprocessing Pipeline")
    st.write("Apply comprehensive preprocessing steps to clean and prepare your dataset for machine learning.")

    # Initialize processed dataframe
    processed_df = df.copy()
    preprocessing_steps = []

    # Step 1: Handle Missing Values
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>Step 1: Handle Missing Values</h4>", unsafe_allow_html=True)

    missing_strategy = st.selectbox(
        "Choose missing value handling strategy:",
        ["Keep as is", "Drop rows with missing values", "Drop columns with missing values",
         "Fill with mean (numeric)", "Fill with median (numeric)", "Fill with mode", "Forward fill", "Custom fill"],
        key="missing_strategy"
    )

    if missing_strategy != "Keep as is":
        if missing_strategy == "Drop rows with missing values":
            initial_rows = processed_df.shape[0]
            processed_df = processed_df.dropna().reset_index(drop=True)
            preprocessing_steps.append(f"Dropped {initial_rows - processed_df.shape[0]} rows with missing values")

        elif missing_strategy == "Drop columns with missing values":
            initial_cols = processed_df.shape[1]
            processed_df = processed_df.dropna(axis=1)
            preprocessing_steps.append(f"Dropped {initial_cols - processed_df.shape[1]} columns with missing values")

        elif missing_strategy == "Fill with mean (numeric)":
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if processed_df[col].isnull().sum() > 0:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            preprocessing_steps.append("Filled numeric columns with mean values")

        elif missing_strategy == "Fill with median (numeric)":
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if processed_df[col].isnull().sum() > 0:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            preprocessing_steps.append("Filled numeric columns with median values")

        elif missing_strategy == "Fill with mode":
            for col in processed_df.columns:
                if processed_df[col].isnull().sum() > 0:
                    mode_val = processed_df[col].mode()
                    if len(mode_val) > 0:
                        processed_df[col] = processed_df[col].fillna(mode_val[0])
            preprocessing_steps.append("Applied forward fill")

        elif missing_strategy == "Custom fill":
            fill_value = st.text_input("Enter custom fill value:", "0", key="custom_fill")
            try:
                processed_df = processed_df.fillna(fill_value)
                preprocessing_steps.append(f"Filled missing values with: {fill_value}")
            except:
                st.error("Invalid fill value")

    # Step 2: Handle Outliers
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>Step 2: Handle Outliers</h4>", unsafe_allow_html=True)

    outlier_method = st.selectbox(
        "Choose outlier handling method:",
        ["Keep as is", "Remove outliers (IQR method)", "Cap outliers (IQR method)"],
        key="outlier_method"
    )

    if outlier_method != "Keep as is":
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_cols = st.multiselect(
                "Select numeric columns to handle outliers:",
                numeric_cols.tolist(),
                default=numeric_cols[:3].tolist(),
                key="outlier_cols"
            )

            if outlier_cols:
                for col in outlier_cols:
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    if outlier_method == "Remove outliers (IQR method)":
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        initial_rows = processed_df.shape[0]
                        processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
                        removed = initial_rows - processed_df.shape[0]
                        preprocessing_steps.append(f"Removed {removed} outliers from {col}")

                    elif outlier_method == "Cap outliers (IQR method)":
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        processed_df[col] = np.clip(processed_df[col], lower_bound, upper_bound)
                        preprocessing_steps.append(f"Capped outliers in {col}")

                processed_df = processed_df.reset_index(drop=True)

    # Step 3: Encode Categorical Variables
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>Step 3: Encode Categorical Variables</h4>", unsafe_allow_html=True)

    categorical_cols = processed_df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_cols) > 0:
        encoding_method = st.selectbox(
            "Choose categorical encoding method:",
            ["Keep as is", "Label Encoding", "One-Hot Encoding"],
            key="encoding_method"
        )

        if encoding_method != "Keep as is":
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                for col in categorical_cols:
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                preprocessing_steps.append("Applied label encoding to categorical columns")

            elif encoding_method == "One-Hot Encoding":
                processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
                preprocessing_steps.append("Applied one-hot encoding to categorical columns")
    else:
        st.info("No categorical columns found for encoding.")

    # Step 4: Feature Scaling
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>Step 4: Feature Scaling</h4>", unsafe_allow_html=True)

    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaling_method = st.selectbox(
            "Choose scaling method:",
            ["No scaling", "Standard Scaling (Z-score)", "Min-Max Scaling", "Robust Scaling"],
            key="scaling_method"
        )

        if scaling_method != "No scaling":
            scaler_cols = st.multiselect(
                "Select columns to scale:",
                numeric_cols.tolist(),
                default=numeric_cols.tolist(),
                key="scaler_cols"
            )

            if scaler_cols:
                if scaling_method == "Standard Scaling (Z-score)":
                    scaler = StandardScaler()
                    processed_df[scaler_cols] = scaler.fit_transform(processed_df[scaler_cols])
                    preprocessing_steps.append("Applied standard scaling")

                elif scaling_method == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                    processed_df[scaler_cols] = scaler.fit_transform(processed_df[scaler_cols])
                    preprocessing_steps.append("Applied min-max scaling")

                elif scaling_method == "Robust Scaling":
                    scaler = RobustScaler()
                    processed_df[scaler_cols] = scaler.fit_transform(processed_df[scaler_cols])
                    preprocessing_steps.append("Applied robust scaling")
    else:
        st.info("No numeric columns found for scaling.")

    # Step 5: Feature Selection (Optional)
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>Step 5: Feature Selection (Optional)</h4>", unsafe_allow_html=True)

    if len(processed_df.columns) > 1:
        do_feature_selection = st.checkbox("Perform feature selection", key="feature_selection")

        if do_feature_selection:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # Assume last column is target if not specified
                # Create formatted column names with data types
                all_cols_for_fs = processed_df.columns.tolist()
                formatted_cols_for_fs = [f"{col} ({str(processed_df[col].dtype)})" for col in all_cols_for_fs]
                col_dtype_map_fs = {f"{col} ({str(processed_df[col].dtype)})": col for col in all_cols_for_fs}
                
                default_target_fs = all_cols_for_fs[-1]
                default_target_fs_formatted = f"{default_target_fs} ({str(processed_df[default_target_fs].dtype)})"
                
                target_col_formatted = st.selectbox(
                    "Select target column for feature selection:",
                    formatted_cols_for_fs,
                    index=formatted_cols_for_fs.index(default_target_fs_formatted) if default_target_fs_formatted in formatted_cols_for_fs else len(formatted_cols_for_fs)-1,
                    key="target_col_fs"
                )
                target_col = col_dtype_map_fs[target_col_formatted]

                feature_cols = [col for col in numeric_cols if col != target_col]
                if len(feature_cols) > 1:
                    k_features = st.slider(
                        "Number of features to select:",
                        min_value=1,
                        max_value=len(feature_cols),
                        value=min(5, len(feature_cols)),
                        key="k_features"
                    )

                    if st.button("Apply Feature Selection", key="apply_fs"):
                        X = processed_df[feature_cols]
                        y = processed_df[target_col]

                        # Choose scoring function based on target type
                        if processed_df[target_col].dtype == 'object' or len(processed_df[target_col].unique()) < 20:
                            score_func = f_classif
                        else:
                            score_func = f_regression

                        selector = SelectKBest(score_func=score_func, k=k_features)
                        X_selected = selector.fit_transform(X, y)

                        selected_features = X.columns[selector.get_support()].tolist()
                        processed_df = processed_df[selected_features + [target_col]]
                        preprocessing_steps.append(f"Selected top {k_features} features: {', '.join(selected_features)}")

    # Display Results
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>Preprocessing Results</h4>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Rows", processed_df.shape[0])
    with col2:
        st.metric("Final Columns", processed_df.shape[1])
    with col3:
        st.metric("Preprocessing Steps", len(preprocessing_steps))

    if preprocessing_steps:
        st.markdown("<h5 style='color: #764ba2;'>Applied Steps:</h5>", unsafe_allow_html=True)
        for step in preprocessing_steps:
            st.markdown(f"• {step}")

    st.markdown("<h5 style='color: #667eea;'>Processed Dataset Preview</h5>", unsafe_allow_html=True)
    st.dataframe(processed_df.head(20), use_container_width=True)

    # Download processed data
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>Download Processed Dataset</h4>", unsafe_allow_html=True)
    download_csv(processed_df, filename="fully_preprocessed_data.csv")


def render_missing_values(df):
    st.subheader("Handle Missing Values")
    st.write("Analyze and manage missing data in your dataset with multiple strategies.")
    
    # Missing value overview
    col1, col2, col3, col4 = st.columns(4)
    missing_vals = df.isnull().sum().sum()
    missing_percent = (missing_vals / (df.shape[0] * df.shape[1])) * 100
    
    with col1:
        st.metric("Total Missing Cells", missing_vals)
    with col2:
        st.metric("Missing %", f"{missing_percent:.2f}%")
    with col3:
        st.metric("Complete Rows", (df.dropna().shape[0]))
    with col4:
        st.metric("Affected Columns", df.columns[df.isnull().any()].shape[0])
    
    st.markdown("---")
    
    # Missing value visualization
    st.markdown("<h4 style='color: #667eea;'>Missing Value Pattern</h4>", unsafe_allow_html=True)
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
            st.success("No missing values detected!")
    
    with col1:
        st.markdown("<h4 style='color: #764ba2;'>Missing Value Percentage</h4>", unsafe_allow_html=True)
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
    st.markdown("<h4 style='color: #667eea;'>Choose Handling Strategy</h4>", unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    if not cols_with_missing:
        st.info("No columns with missing values found!")
        return
    
    strategy = st.selectbox("Select a strategy", [
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
        st.markdown("<h4 style='color: #764ba2;'>Preview of Processed Data</h4>", unsafe_allow_html=True)
        
        if strategy == "Drop rows with missing values":
            processed_df = df.dropna().reset_index(drop=True)
            st.info(f"Removed {df.shape[0] - processed_df.shape[0]} rows with missing values")
            
        elif strategy == "Fill with mean (numeric columns)":
            processed_df = df.copy()
            for col in numeric_cols:
                if processed_df[col].isnull().sum() > 0:
                    processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            st.info("Filled numeric columns with mean values")
            
        elif strategy == "Fill with median (numeric columns)":
            processed_df = df.copy()
            for col in numeric_cols:
                if processed_df[col].isnull().sum() > 0:
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
            st.info("Filled numeric columns with median values")
            
        elif strategy == "Fill with mode (most frequent)":
            processed_df = df.copy()
            for col in df.columns:
                if processed_df[col].isnull().sum() > 0:
                    mode_val = processed_df[col].mode()
                    if len(mode_val) > 0:
                        processed_df[col].fillna(mode_val[0], inplace=True)
            st.info("Filled columns with mode (most frequent value)")
            
        elif strategy == "Forward fill":
            processed_df = df.fillna(method='ffill').fillna(method='bfill')
            st.info("Applied forward fill and then backward fill")
            
        elif strategy == "Backward fill":
            processed_df = df.fillna(method='bfill').fillna(method='ffill')
            st.info("Applied backward fill and then forward fill")
            
        elif strategy == "Linear interpolation (numeric only)":
            processed_df = df.copy()
            for col in numeric_cols:
                processed_df[col] = processed_df[col].interpolate(method='linear')
            st.info("Applied linear interpolation to numeric columns")
            
        elif strategy == "Custom value fill":
            fill_value = st.text_input("Enter value to fill missing data with:", "0")
            try:
                processed_df = df.fillna(fill_value)
                st.info(f"Filled missing values with: {fill_value}")
            except:
                st.error("Invalid fill value")
                return
        
        st.dataframe(processed_df.head(15), use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='color: #764ba2; text-align: center;'>Statistics</h4>", unsafe_allow_html=True)
        remaining_missing = processed_df.isnull().sum().sum()
        st.metric("Remaining Missing Values", remaining_missing)
        st.metric("Rows After Processing", processed_df.shape[0])
        st.metric("Columns", processed_df.shape[1])
    
    st.markdown("---")
    
    # Download processed data
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color: #667eea;'>Export Processed Data</h4>", unsafe_allow_html=True)
        download_csv(processed_df, filename="handled_missing_values.csv")
    
    with col2:
        if st.button("Show Processing Summary"):
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea15, #764ba215); padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;'>
            <b>Processing Summary:</b><br>
            Original rows: <b>{df.shape[0]}</b><br>
            Final rows: <b>{processed_df.shape[0]}</b><br>
            Rows removed: <b>{df.shape[0] - processed_df.shape[0]}</b><br>
            Original missing values: <b>{df.isnull().sum().sum()}</b><br>
            Final missing values: <b>{processed_df.isnull().sum().sum()}</b>
            </div>
            """, unsafe_allow_html=True)




def plot_decision_tree_viz(model, feature_names, class_names=None, max_depth=3):
    """Plot decision tree visualization"""
    try:
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.patch.set_facecolor('#FAFBFF')
        ax.set_facecolor('#F5F7FA')
        
        plot_tree(model, 
                 feature_names=feature_names,
                 class_names=class_names,
                 filled=True,
                 rounded=True,
                 fontsize=9,
                 ax=ax,
                 max_depth=max_depth)
        
        st.pyplot(fig)
        return True
    except Exception as e:
        st.warning(f"Could not visualize tree: {str(e)}")
        return False


def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based and ensemble models"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#FAFBFF')
            ax.set_facecolor('#F5F7FA')
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
            ax.bar(range(len(indices)), importances[indices], color=colors, edgecolor='white', linewidth=2)
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            ax.set_xlabel('Features', fontsize=11, fontweight='bold', color='#424242')
            ax.set_ylabel('Importance', fontsize=11, fontweight='bold', color='#424242')
            ax.set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold', color='#667eea')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='both', labelcolor='#424242')
            
            st.pyplot(fig)
            return True
    except Exception as e:
        st.warning(f"Could not plot feature importance: {str(e)}")
    return False


def convert_param_values(params_dict):
    """Convert parameter values to appropriate types for model instantiation"""
    converted = {}
    for key, value in params_dict.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            converted[key] = value
        elif isinstance(value, np.integer):
            converted[key] = int(value)
        elif isinstance(value, np.floating):
            converted[key] = float(value)
        else:
            converted[key] = value
    return converted


def render_models(df):
    st.subheader("Machine Learning Models")
    st.write("Train and compare multiple ML models on your dataset. Select features and target column to get started.")
    
    # Data Preparation
    st.markdown("<h4 style='color: #667eea;'>Model Configuration</h4>", unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    # Feature and Target Selection with auto-defaults
    col1, col2 = st.columns(2)

    # Create formatted column names with data types
    formatted_cols = [f"{col} ({str(df[col].dtype)})" for col in all_cols]
    col_dtype_map = {f"{col} ({str(df[col].dtype)})": col for col in all_cols}

    with col1:
        st.write("**Select Target Column:**")
        default_target = numeric_cols[-1] if numeric_cols else all_cols[0]
        default_target_formatted = f"{default_target} ({str(df[default_target].dtype)})"
        target_col_formatted = st.selectbox(
            "Choose target column to predict",
            formatted_cols,
            index=formatted_cols.index(default_target_formatted) if default_target_formatted in formatted_cols else 0,
            key="target_selection"
        )
        target_col = col_dtype_map[target_col_formatted]

    with col2:
        st.write("**Select Features (Predictors):**")
        available_features = [col for col in all_cols if col != target_col]
        formatted_available_features = [f"{col} ({str(df[col].dtype)})" for col in available_features]
        default_features = [col for col in available_features if col in numeric_cols][:min(len(available_features), 5)]
        if not default_features and available_features:
            default_features = available_features[:min(len(available_features), 5)]
        default_features_formatted = [f"{col} ({str(df[col].dtype)})" for col in default_features]
        selected_features_formatted = st.multiselect(
            "Choose columns as features",
            formatted_available_features,
            default=default_features_formatted,
            key="feature_selection"
        )
        selected_features = [col_dtype_map[formatted_col] for formatted_col in selected_features_formatted]
    
    if not selected_features:
        st.error("Please select at least one feature column")
        return
    
    if target_col in selected_features:
        st.error("Target column cannot be a feature column")
        return
    
    # INSIGHT SECTION
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>Dataset Insights & Analysis Preview</h4>", unsafe_allow_html=True)
    
    # Create insight columns
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("<h5 style='color: #667eea;'>Selected Features Overview</h5>", unsafe_allow_html=True)
        
        # Feature insights
        feature_insights = []
        for col in selected_features:
            col_data = df[col].dropna()
            feature_info = {
                'Feature': col,
                'Type': str(df[col].dtype),
                'Range': f"{col_data.min():.2f} - {col_data.max():.2f}" if col_data.dtype in ['int64', 'float64'] else f"{col_data.nunique()} unique values",
                'Missing': f"{df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)",
                'Samples': len(col_data)
            }
            feature_insights.append(feature_info)
        
        feature_df = pd.DataFrame(feature_insights)
        st.dataframe(feature_df, use_container_width=True)
        
        # Feature correlation preview
        numeric_feature_cols = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_feature_cols) > 1:
            st.markdown("<h6 style='color: #764ba2;'>Feature Correlations</h6>", unsafe_allow_html=True)
            corr_matrix = df[numeric_feature_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#FAFBFF')
            ax.set_facecolor('#F5F7FA')
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       ax=ax, cbar_kws={'shrink': 0.8}, linewidths=0.5,
                       annot_kws={'size': 8})
            ax.set_title('Feature Correlation Matrix', fontsize=10, fontweight='bold', color='#667eea')
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)
    
    with insight_col2:
        st.markdown("<h5 style='color: #764ba2;'>Target Column Analysis</h5>", unsafe_allow_html=True)
        
        # Target insights
        target_data = df[target_col].dropna()
        target_info = {
            'Target': target_col,
            'Type': str(df[target_col].dtype),
            'Missing': f"{df[target_col].isnull().sum()} ({df[target_col].isnull().sum()/len(df)*100:.1f}%)",
            'Samples': len(target_data)
        }
        
        # Detect problem type for insights
        if df[target_col].dtype == 'object':
            problem_type_insight = 'Classification'
            target_info['Classes'] = f"{len(target_data.unique())} classes"
            target_info['Distribution'] = f"Most common: {target_data.mode().iloc[0]}"
        else:
            unique_vals = len(target_data.unique())
            if unique_vals < 20 and df[target_col].dtype in ['int64', 'int32']:
                problem_type_insight = 'Classification'
                target_info['Classes'] = f"{unique_vals} classes"
                target_info['Distribution'] = f"Values: {sorted(target_data.unique())[:5]}{'...' if unique_vals > 5 else ''}"
            else:
                problem_type_insight = 'Regression'
                target_info['Range'] = f"{target_data.min():.2f} - {target_data.max():.2f}"
                target_info['Distribution'] = f"Mean: {target_data.mean():.2f}, Std: {target_data.std():.2f}"
        
        target_info['Problem Type'] = problem_type_insight
        
        # Display target info
        for key, value in target_info.items():
            if key == 'Target':
                st.markdown(f"**{key}:** {value}")
            else:
                st.markdown(f"**{key}:** {value}")
        
        # Target distribution visualization
        st.markdown("<h6 style='color: #667eea;'>Target Distribution</h6>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#FAFBFF')
        ax.set_facecolor('#F5F7FA')
        
        if problem_type_insight == 'Classification':
            if len(target_data.unique()) <= 10:
                target_data.value_counts().plot(kind='bar', ax=ax, color='#5DA9E8', edgecolor='white', linewidth=1)
                ax.set_xlabel('Classes', fontsize=9, fontweight='bold')
            else:
                target_data.value_counts().head(10).plot(kind='bar', ax=ax, color='#5DA9E8', edgecolor='white', linewidth=1)
                ax.set_xlabel('Top 10 Classes', fontsize=9, fontweight='bold')
        else:
            ax.hist(target_data, bins=20, color='#5DA9E8', edgecolor='white', linewidth=1, alpha=0.7)
            ax.set_xlabel('Values', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax.set_title(f'Distribution of {target_col}', fontsize=10, fontweight='bold', color='#667eea')
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    # Expected Results Preview
    st.markdown("---")
    st.markdown("<h5 style='color: #667eea;'>Expected Model Results</h5>", unsafe_allow_html=True)
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%); 
                    padding: 15px; border-radius: 10px; border-left: 4px solid #5DA9E8;'>
            <h6 style='color: #1565C0; margin: 0;'>Problem Type</h6>
            <p style='color: #424242; margin: 5px 0; font-weight: 600;'>{problem_type_insight}</p>
            <small style='color: #666;'>Based on target column analysis</small>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col2:
        models_count = 7 if problem_type_insight == 'Classification' else 6
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FFF3E0 0%, #FCE4EC 100%); 
                    padding: 15px; border-radius: 10px; border-left: 4px solid #FFB347;'>
            <h6 style='color: #E65100; margin: 0;'>Models to Train</h6>
            <p style='color: #424242; margin: 5px 0; font-weight: 600;'>{models_count} ML Models</p>
            <small style='color: #666;'>Auto-selected for {problem_type_insight.lower()}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col3:
        if problem_type_insight == 'Classification':
            expected_output = f"Predict class labels from {len(target_data.unique())} categories"
        else:
            expected_output = f"Predict continuous values (range: {target_data.min():.2f} - {target_data.max():.2f})"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%); 
                    padding: 15px; border-radius: 10px; border-left: 4px solid #66BB6A;'>
            <h6 style='color: #2E7D32; margin: 0;'>Expected Output</h6>
            <p style='color: #424242; margin: 5px 0; font-size: 12px;'>{expected_output}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ADVANCED INSIGHTS SECTION
    st.markdown("---")
    st.markdown("<h4 style='color: #667eea;'>📊 Advanced Dataset Insights & Recommendations</h4>", unsafe_allow_html=True)
    
    # Calculate insights
    numeric_features = [col for col in selected_features if df[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in selected_features if df[col].dtype == 'object']
    valid_samples = len(df.dropna(subset=selected_features + [target_col]))
    missing_pct = (1 - valid_samples / len(df)) * 100
    
    # Feature distribution insights
    insights_col1, insights_col2, insights_col3 = st.columns(3)
    
    with insights_col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); 
                    padding: 12px; border-radius: 8px; border-left: 4px solid #FF6B9D;'>
            <h6 style='color: #C2185B; margin: 0; font-size: 12px;'>🎯 Feature Composition</h6>
            <p style='color: #424242; margin: 8px 0; font-size: 13px;'><b>Numeric:</b> {}<br/><b>Categorical:</b> {}</p>
        </div>
        """.format(len(numeric_features), len(categorical_features)), unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                    padding: 12px; border-radius: 8px; border-left: 4px solid #1976D2;'>
            <h6 style='color: #0D47A1; margin: 0; font-size: 12px;'>⚠️ Data Completeness</h6>
            <p style='color: #424242; margin: 8px 0; font-size: 13px;'><b>Valid Samples:</b> {}/{}<br/><b>Missing:</b> {:.1f}%</p>
        </div>
        """.format(valid_samples, len(df), missing_pct), unsafe_allow_html=True)
    
    with insights_col3:
        # Feature variability
        if numeric_features:
            variance_scores = []
            for col in numeric_features:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    variance_scores.append(col_data.std() / (col_data.mean() + 1e-8))
            avg_cv = np.mean(variance_scores) if variance_scores else 0
            st.markdown("""
            <div style='background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%); 
                        padding: 12px; border-radius: 8px; border-left: 4px solid #7B1FA2;'>
                <h6 style='color: #4A148C; margin: 0; font-size: 12px;'>📈 Feature Variability</h6>
                <p style='color: #424242; margin: 8px 0; font-size: 13px;'><b>Coefficient of Variation:</b> {:.2f}<br/><b>Status:</b> {}</p>
            </div>
            """.format(avg_cv, "High Variance" if avg_cv > 1 else "Moderate" if avg_cv > 0.5 else "Low Variance"), unsafe_allow_html=True)
    
    # Class imbalance detection for classification
    if problem_type_insight == 'Classification':
        st.markdown("<h6 style='color: #667eea;'>⚖️ Class Balance Analysis</h6>", unsafe_allow_html=True)
        class_dist = target_data.value_counts()
        imbalance_ratio = class_dist.max() / class_dist.min()
        
        imbalance_col1, imbalance_col2 = st.columns(2)
        
        with imbalance_col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#FAFBFF')
            ax.set_facecolor('#F5F7FA')
            class_dist.plot(kind='barh', ax=ax, color=['#5DA9E8', '#FFB347', '#66BB6A', '#FF6B9D'][:len(class_dist)])
            ax.set_xlabel('Count', fontweight='bold', fontsize=9)
            ax.set_title('Class Distribution', fontweight='bold', color='#667eea', fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
        
        with imbalance_col2:
            if imbalance_ratio > 3:
                imbalance_status = "⚠️ **Highly Imbalanced** - Consider using class weights or resampling techniques"
                bg_color = "#FFF3E0"
                text_color = "#E65100"
            elif imbalance_ratio > 1.5:
                imbalance_status = "📌 **Moderately Imbalanced** - Monitor metrics closely during evaluation"
                bg_color = "#FFF8E1"
                text_color = "#F57F17"
            else:
                imbalance_status = "✅ **Well Balanced** - Good class distribution"
                bg_color = "#E8F5E9"
                text_color = "#2E7D32"
            
            st.markdown(f"""
            <div style='background: {bg_color}; padding: 12px; border-radius: 8px; border-left: 4px solid {text_color};'>
                <p style='color: {text_color}; margin: 5px 0; font-size: 13px;'>{imbalance_status}</p>
                <p style='color: #424242; margin: 5px 0; font-size: 12px;'><b>Imbalance Ratio:</b> {imbalance_ratio:.2f}:1</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Data quality recommendations
    st.markdown("<h6 style='color: #667eea;'>💡 Data Quality Recommendations</h6>", unsafe_allow_html=True)
    
    recommendations = []
    
    # Missing data check
    if missing_pct > 5:
        recommendations.append(f"🔴 **High Missing Data ({missing_pct:.1f}%)** - Consider imputation or row removal before training")
    elif missing_pct > 0:
        recommendations.append(f"🟡 **Minor Missing Data ({missing_pct:.1f}%)** - Address missing values for optimal performance")
    else:
        recommendations.append("✅ **No Missing Data** - Dataset is complete")
    
    # Categorical encoding check
    if categorical_features:
        recommendations.append(f"⚠️ **Categorical Features Detected** - Ensure proper encoding (Label/One-Hot) in preprocessing")
    
    # Feature scaling check
    if numeric_features and len(numeric_features) > 1:
        feature_scales = []
        for col in numeric_features:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                scale = col_data.max() - col_data.min()
                feature_scales.append(scale)
        if max(feature_scales) / (min(feature_scales) + 1e-8) > 100:
            recommendations.append("🔴 **Wide Feature Scale Range** - Consider applying scaling/normalization")
    
    # Sample size check
    if valid_samples < 100:
        recommendations.append("🔴 **Small Dataset** - Consider cross-validation; risk of overfitting")
    elif valid_samples < 1000:
        recommendations.append("🟡 **Moderate Dataset Size** - Stratified k-fold cross-validation recommended")
    else:
        recommendations.append("✅ **Adequate Sample Size** - Sufficient data for model training")
    
    # Feature-target correlation for regression
    if problem_type_insight == 'Regression' and numeric_features:
        correlations = [df[col].corr(df[target_col]) for col in numeric_features]
        avg_corr = np.mean(np.abs(correlations))
        if avg_corr < 0.2:
            recommendations.append("⚠️ **Low Feature-Target Correlation** - Selected features may have weak predictive power")
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}", unsafe_allow_html=True)
    
    # Model selection guidance
    st.markdown("<h6 style='color: #667eea;'>🤖 Model Selection Guidance</h6>", unsafe_allow_html=True)
    
    if problem_type_insight == 'Classification':
        guidance_text = """
        - **Logistic Regression**: Fast baseline, good for linear relationships
        - **Decision Tree**: Handles non-linear patterns, interpretable
        - **Random Forest**: Excellent for feature interactions, robust to outliers
        - **Gradient Boosting**: High performance, but requires careful tuning
        - **KNN**: Good for local patterns, sensitive to feature scaling
        - **SVM**: Powerful for complex boundaries, requires scaling
        - **Naive Bayes**: Fast, works well with text/categorical data
        """
    else:
        guidance_text = """
        - **Linear Regression**: Simple baseline, interpretable
        - **Decision Tree Regressor**: Captures non-linear patterns
        - **Random Forest Regressor**: Handles feature interactions well
        - **Gradient Boosting Regressor**: High predictive power, needs tuning
        - **KNN Regressor**: Good for local approximations
        - **SVR**: Effective for complex non-linear relationships
        """
    
    st.markdown(guidance_text)
    
    # Quick stats summary
    st.markdown("---")
    st.markdown("<h6 style='color: #667eea;'>Quick Dataset Summary</h6>", unsafe_allow_html=True)
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Features Selected", len(selected_features))
    
    with summary_col2:
        st.metric("Valid Samples", valid_samples)
    
    with summary_col3:
        if problem_type_insight == 'Classification':
            st.metric("Target Classes", len(target_data.unique()))
        else:
            st.metric("Target Range", f"{target_data.min():.1f} - {target_data.max():.1f}")
    
    with summary_col4:
        st.metric("Data Quality", f"{(1 - missing_pct/100) * 100:.1f}%")
    
    st.markdown("---")
    
    try:
        # Prepare data
        X = df[selected_features].copy()
        y = df[target_col].copy()

        # Remove rows with missing values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)

        if len(X) < 10:
            st.error(f"Not enough valid data. Found {len(X)} valid samples, need at least 10.")
            return

        # Encode categorical features automatically
        categorical_feature_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_feature_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Detect problem type
        if y.dtype == 'object':
            problem_type = 'classification'
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
            target_classes = le.classes_
            n_classes = len(target_classes)
        else:
            # Check if it's classification (integer with few unique values) or regression
            unique_vals = len(y.unique())
            if unique_vals < 20 and y.dtype in ['int64', 'int32']:
                problem_type = 'classification'
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                target_classes = le.classes_
                n_classes = len(target_classes)
            else:
                problem_type = 'regression'
                y_encoded = y
                target_classes = None
                n_classes = None
        
        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", len(X))
        with col2:
            st.metric("Features", len(selected_features))
        with col3:
            if problem_type == 'classification':
                st.metric("Classes", n_classes)
            else:
                st.metric("Range", f"{y.min():.2f} - {y.max():.2f}")
        with col4:
            st.metric("Valid Data %", f"{(len(X)/len(df))*100:.1f}%")
        
        st.info(f"Detected: **{problem_type.upper()}** problem with target '{target_col}'")
        st.markdown("---")
        
        # Train-test split
        col1, col2 = st.columns([2, 1])
        with col1:
            test_size = st.slider("Train-Test Split (Test %):", 10, 40, 20, key="test_size") / 100
        with col2:
            random_seed = st.number_input("Random Seed:", value=42, min_value=0, key="random_seed")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_seed
        )
        
        # Scale features for better model performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.markdown("---")
        st.markdown("<h4 style='color: #667eea;'>Training Models...</h4>", unsafe_allow_html=True)
        
        # Define models based on problem type
        if problem_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_seed),
                'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=random_seed),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_seed),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_seed),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Support Vector Machine': SVC(kernel='rbf', random_state=random_seed),
                'Naive Bayes': GaussianNB(),
            }
        else:  # regression
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=random_seed),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_seed),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_seed),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
                'Support Vector Machine': SVR(kernel='rbf'),
            }
        
        # Train models
        results = []
        predictions = {}
        trained_models = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}... ({idx+1}/{len(models)})")
            try:
                # Use scaled features for most models
                if name in ['Support Vector Machine', 'K-Nearest Neighbors', 'Logistic Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                trained_models[name] = model
                predictions[name] = y_pred
                
                # Calculate metrics based on problem type
                if problem_type == 'classification':
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
                        'Score': accuracy,
                    })
                else:  # regression
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    results.append({
                        'Model': name,
                        'R² Score': f"{r2:.4f}",
                        'MAE': f"{mae:.4f}",
                        'RMSE': f"{rmse:.4f}",
                        'MSE': f"{mse:.4f}",
                        'Score': r2,
                    })
                    
            except Exception as e:
                st.warning(f"{name} training failed: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(models))
        
        status_text.empty()
        progress_bar.empty()
        
        if not results:
            st.error("No models trained successfully")
            return
        
        results_df = pd.DataFrame(results)
        st.success(f"Successfully trained {len(results)} models!")
        
        st.markdown("---")
        st.markdown("<h4 style='color: #764ba2;'>Model Performance Comparison</h4>", unsafe_allow_html=True)
        
        # Show results table
        display_df = results_df.drop('Score', axis=1)
        st.dataframe(display_df, use_container_width=True)
        
        st.markdown("---")
        
        # Find best model
        best_idx = results_df['Score'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, 'Score']
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if problem_type == 'classification':
                st.markdown("<h5 style='color: #667eea;'>Accuracy Comparison</h5>", unsafe_allow_html=True)
                metric_col = 'Accuracy'
            else:
                st.markdown("<h5 style='color: #667eea;'>R² Score Comparison</h5>", unsafe_allow_html=True)
                metric_col = 'R² Score'
            
            # Parse scores for plotting
            plot_data = results_df.sort_values('Score', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#FAFBFF')
            ax.set_facecolor('#F5F7FA')
            
            colors = ['#66BB6A' if score == plot_data['Score'].max() else '#5DA9E8' 
                     for score in plot_data['Score']]
            
            bars = ax.barh(plot_data['Model'], plot_data['Score'], color=colors, edgecolor='white', linewidth=2)
            ax.set_xlabel('Score', fontsize=11, fontweight='bold', color='#424242')
            ax.set_title(f'Model {metric_col} Comparison', fontsize=12, fontweight='bold', color='#667eea')
            if problem_type == 'classification':
                ax.set_xlim([0, 1.05])
            ax.tick_params(axis='both', labelcolor='#424242')
            ax.grid(axis='x', alpha=0.3, color='#E8EAEF', linestyle='--')
            
            # Add value labels
            for i, (idx, row) in enumerate(plot_data.iterrows()):
                if problem_type == 'classification':
                    label = f"{row['Score']*100:.1f}%"
                else:
                    label = f"{row['Score']:.4f}"
                ax.text(row['Score'] + 0.02, i, label, va='center', fontweight='bold', 
                       color='#424242', fontsize=10)
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("<h5 style='color: #764ba2;'>Best Performing Model</h5>", unsafe_allow_html=True)
            
            if problem_type == 'classification':
                score_label = f"{best_score*100:.2f}% Accuracy"
            else:
                score_label = f"{best_score:.4f} R² Score"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #66BB6A 0%, #4CAF50 100%); 
                        padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                <h3 style='margin: 0; font-size: 24px;'>{best_model_name}</h3>
                <p style='margin: 10px 0 0 0; font-size: 20px; font-weight: 600;'>{score_label}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h5 style='color: #667eea; margin-top: 20px;'>Best Model Metrics</h5>", 
                       unsafe_allow_html=True)
            best_row = results_df.loc[best_idx]
            
            metrics_cols = st.columns(2)
            metrics_idx = 0
            for col_name in best_row.index:
                if col_name not in ['Model', 'Score']:
                    with metrics_cols[metrics_idx % 2]:
                        st.metric(col_name, best_row[col_name])
                    metrics_idx += 1
        
        st.markdown("---")
        
        # Detailed analysis for best model
        st.markdown("<h4 style='color: #667eea;'>Detailed Analysis - Best Model</h4>", unsafe_allow_html=True)
        
        if problem_type == 'classification' and n_classes <= 10:
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                st.markdown("<h5 style='color: #667eea;'>Confusion Matrix</h5>", unsafe_allow_html=True)
                best_predictions = predictions[best_model_name]
                cm = confusion_matrix(y_test, best_predictions)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#FAFBFF')
                ax.set_facecolor('#F5F7FA')
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=True,
                           xticklabels=target_classes, yticklabels=target_classes,
                           ax=ax, cbar_kws={'label': 'Count'}, linewidths=2, linecolor='white',
                           annot_kws={'size': 11, 'weight': 'bold', 'color': '#424242'})
                
                ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', color='#424242')
                ax.set_ylabel('True Label', fontsize=11, fontweight='bold', color='#424242')
                ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold', 
                           color='#667eea')
                ax.tick_params(axis='both', labelcolor='#424242')
                
                st.pyplot(fig)
            
            with col2:
                st.markdown("<h5 style='color: #764ba2;'>Classification Report</h5>", unsafe_allow_html=True)
                best_predictions = predictions[best_model_name]
                report = classification_report(y_test, best_predictions, target_names=target_classes, 
                                             output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)
        
        elif problem_type == 'regression':
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h5 style='color: #667eea;'>Predicted vs Actual</h5>", unsafe_allow_html=True)
                best_predictions = predictions[best_model_name]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#FAFBFF')
                ax.set_facecolor('#F5F7FA')
                
                ax.scatter(y_test, best_predictions, alpha=0.6, color='#5DA9E8', edgecolor='white', linewidth=1, s=50)
                min_val = min(y_test.min(), best_predictions.min())
                max_val = max(y_test.max(), best_predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                
                ax.set_xlabel('Actual Values', fontsize=11, fontweight='bold', color='#424242')
                ax.set_ylabel('Predicted Values', fontsize=11, fontweight='bold', color='#424242')
                ax.set_title(f'Predicted vs Actual - {best_model_name}', fontsize=12, fontweight='bold', 
                           color='#667eea')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelcolor='#424242')
                
                st.pyplot(fig)
            
            with col2:
                st.markdown("<h5 style='color: #764ba2;'>Residuals Analysis</h5>", unsafe_allow_html=True)
                best_predictions = predictions[best_model_name]
                residuals = y_test - best_predictions
                
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#FAFBFF')
                ax.set_facecolor('#F5F7FA')
                
                ax.hist(residuals, bins=20, color='#5DA9E8', edgecolor='white', linewidth=1.5, alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                
                ax.set_xlabel('Residuals', fontsize=11, fontweight='bold', color='#424242')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold', color='#424242')
                ax.set_title(f'Residuals Distribution - {best_model_name}', fontsize=12, fontweight='bold', 
                           color='#667eea')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='both', labelcolor='#424242')
                
                st.pyplot(fig)
        
        # Model comparison details
        st.markdown("---")
        st.markdown("<h4 style='color: #764ba2;'>All Models Summary</h4>", unsafe_allow_html=True)
        
        with st.expander("Detailed Results for All Models"):
            st.dataframe(results_df.drop('Score', axis=1), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Total Models Trained:** {len(results)}")
                st.markdown(f"**Best Model:** {best_model_name}")
            with col2:
                st.markdown(f"**Training Data Size:** {len(X_train)}")
                st.markdown(f"**Testing Data Size:** {len(X_test)}")
        
        # ===== MANUAL MODEL SELECTION SECTION =====
        st.markdown("---")
        st.markdown("<h4 style='color: #667eea;'>Select Individual Model for Detailed Analysis</h4>", unsafe_allow_html=True)
        st.markdown("Choose a specific model below to view its detailed metrics and performance analysis.")
        
        # Create tabs for automatic vs manual analysis
        tab1, tab2 = st.tabs(["Automatic (All Models)", "Manual (Single Model)"])
        
        with tab2:
            # Manual model selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_model_name = st.selectbox(
                    "Select a Model to Analyze:",
                    options=results_df['Model'].tolist(),
                    index=0,
                    key="manual_model_selection"
                )
            
            with col2:
                st.markdown("<p style='color: #667eea; margin-top: 30px; text-align: center;'><b>Model Performance</b></p>", unsafe_allow_html=True)
            
            if selected_model_name:
                # Get model metrics
                selected_model_row = results_df[results_df['Model'] == selected_model_name].iloc[0]
                selected_model_obj = trained_models[selected_model_name]
                selected_predictions = predictions[selected_model_name]
                
                st.markdown("---")
                
                # Display selected model metrics in columns
                metric_cols = st.columns(4) if problem_type == 'classification' else st.columns(4)
                
                if problem_type == 'classification':
                    # Classification metrics
                    with metric_cols[0]:
                        accuracy = accuracy_score(y_test, selected_predictions)
                        st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    
                    with metric_cols[1]:
                        precision = precision_score(y_test, selected_predictions, average='weighted', zero_division=0)
                        st.metric("Precision", f"{precision*100:.2f}%")
                    
                    with metric_cols[2]:
                        recall = recall_score(y_test, selected_predictions, average='weighted', zero_division=0)
                        st.metric("Recall", f"{recall*100:.2f}%")
                    
                    with metric_cols[3]:
                        f1 = f1_score(y_test, selected_predictions, average='weighted', zero_division=0)
                        st.metric("F1-Score", f"{f1*100:.2f}%")
                
                else:
                    # Regression metrics
                    with metric_cols[0]:
                        r2 = r2_score(y_test, selected_predictions)
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    with metric_cols[1]:
                        mae = mean_absolute_error(y_test, selected_predictions)
                        st.metric("MAE", f"{mae:.4f}")
                    
                    with metric_cols[2]:
                        mse = mean_squared_error(y_test, selected_predictions)
                        st.metric("MSE", f"{mse:.4f}")
                    
                    with metric_cols[3]:
                        rmse = np.sqrt(mse)
                        st.metric("RMSE", f"{rmse:.4f}")
                
                st.markdown("---")
                
                # ===== PARAMETER ADJUSTMENT SECTION =====
                with st.expander("🔧 Customize Model Parameters & Retrain", expanded=False):
                    st.markdown("<p style='color: #667eea; font-weight: 600;'>Modify parameters below and click 'Retrain Model' to see how changes affect performance</p>", unsafe_allow_html=True)
                    
                    adjusted_params = {}
                    current_params = selected_model_obj.get_params()
                    
                    # Create parameter adjustment widgets based on model type
                    param_cols = st.columns(2)
                    param_col_idx = 0
                    
                    for param_name, param_value in current_params.items():
                        # Skip certain parameters
                        if param_name in ['random_state', 'n_jobs', 'verbose', 'warm_start']:
                            continue
                        
                        with param_cols[param_col_idx % 2]:
                            if isinstance(param_value, bool):
                                adjusted_params[param_name] = st.checkbox(
                                    f"{param_name}",
                                    value=param_value,
                                    key=f"param_{param_name}_{selected_model_name}"
                                )
                            elif isinstance(param_value, int):
                                # Set reasonable ranges for integer parameters
                                if 'depth' in param_name.lower():
                                    min_val, max_val = 1, 20
                                elif 'neighbors' in param_name.lower():
                                    min_val, max_val = 1, 20
                                elif 'estimators' in param_name.lower():
                                    min_val, max_val = 10, 500
                                else:
                                    min_val, max_val = max(1, param_value - 10), param_value + 10
                                
                                adjusted_params[param_name] = st.slider(
                                    f"{param_name}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=param_value,
                                    key=f"param_{param_name}_{selected_model_name}"
                                )
                            elif isinstance(param_value, float):
                                adjusted_params[param_name] = st.slider(
                                    f"{param_name}",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=param_value,
                                    step=0.01,
                                    key=f"param_{param_name}_{selected_model_name}"
                                )
                            elif isinstance(param_value, str):
                                options = {
                                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                                    'criterion': ['gini', 'entropy'],
                                    'splitter': ['best', 'random'],
                                    'loss': ['linear', 'square', 'exponential', 'squared_error'],
                                }
                                if param_name in options:
                                    adjusted_params[param_name] = st.selectbox(
                                        f"{param_name}",
                                        options=options[param_name],
                                        index=options[param_name].index(param_value) if param_value in options[param_name] else 0,
                                        key=f"param_{param_name}_{selected_model_name}"
                                    )
                                else:
                                    adjusted_params[param_name] = param_value
                            else:
                                adjusted_params[param_name] = param_value
                            param_col_idx += 1
                    
                    # Retrain button
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("🔄 Retrain Model", key=f"retrain_{selected_model_name}"):
                            # Convert to appropriate types
                            adjusted_params_converted = convert_param_values(adjusted_params)
                            
                            try:
                                # Create new model with adjusted parameters
                                if selected_model_name == 'Logistic Regression':
                                    new_model = LogisticRegression(max_iter=1000, random_state=42, **adjusted_params_converted)
                                elif selected_model_name == 'Decision Tree':
                                    if problem_type == 'classification':
                                        new_model = DecisionTreeClassifier(random_state=42, **adjusted_params_converted)
                                    else:
                                        new_model = DecisionTreeRegressor(random_state=42, **adjusted_params_converted)
                                elif selected_model_name == 'Random Forest':
                                    if problem_type == 'classification':
                                        new_model = RandomForestClassifier(random_state=42, **adjusted_params_converted)
                                    else:
                                        new_model = RandomForestRegressor(random_state=42, **adjusted_params_converted)
                                elif selected_model_name == 'Gradient Boosting':
                                    if problem_type == 'classification':
                                        new_model = GradientBoostingClassifier(random_state=42, **adjusted_params_converted)
                                    else:
                                        new_model = GradientBoostingRegressor(random_state=42, **adjusted_params_converted)
                                elif selected_model_name == 'K-Nearest Neighbors':
                                    if problem_type == 'classification':
                                        new_model = KNeighborsClassifier(**adjusted_params_converted)
                                    else:
                                        new_model = KNeighborsRegressor(**adjusted_params_converted)
                                elif selected_model_name == 'Support Vector Machine':
                                    if problem_type == 'classification':
                                        new_model = SVC(random_state=42, **adjusted_params_converted)
                                    else:
                                        new_model = SVR(**adjusted_params_converted)
                                elif selected_model_name == 'Naive Bayes':
                                    new_model = GaussianNB(**adjusted_params_converted)
                                elif selected_model_name == 'Linear Regression':
                                    new_model = LinearRegression(**adjusted_params_converted)
                                
                                # Train with scaled features if needed
                                if selected_model_name in ['Support Vector Machine', 'K-Nearest Neighbors', 'Logistic Regression']:
                                    new_model.fit(X_train_scaled, y_train)
                                    new_predictions = new_model.predict(X_test_scaled)
                                else:
                                    new_model.fit(X_train, y_train)
                                    new_predictions = new_model.predict(X_test)
                                
                                # Calculate new metrics
                                st.success("✅ Model retrained successfully!")
                                
                                # Display new metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                if problem_type == 'classification':
                                    new_accuracy = accuracy_score(y_test, new_predictions)
                                    new_precision = precision_score(y_test, new_predictions, average='weighted', zero_division=0)
                                    new_recall = recall_score(y_test, new_predictions, average='weighted', zero_division=0)
                                    new_f1 = f1_score(y_test, new_predictions, average='weighted', zero_division=0)
                                    
                                    with col1:
                                        old_accuracy = accuracy_score(y_test, selected_predictions)
                                        st.metric("Accuracy (Old)", f"{old_accuracy*100:.2f}%")
                                    with col2:
                                        st.metric("Accuracy (New)", f"{new_accuracy*100:.2f}%", 
                                                 delta=f"{(new_accuracy-old_accuracy)*100:.2f}%")
                                    with col3:
                                        st.metric("F1-Score (Old)", f"{f1*100:.2f}%")
                                    with col4:
                                        st.metric("F1-Score (New)", f"{new_f1*100:.2f}%",
                                                 delta=f"{(new_f1-f1)*100:.2f}%")
                                else:
                                    new_r2 = r2_score(y_test, new_predictions)
                                    new_mae = mean_absolute_error(y_test, new_predictions)
                                    new_mse = mean_squared_error(y_test, new_predictions)
                                    new_rmse = np.sqrt(new_mse)
                                    
                                    with col1:
                                        old_r2 = r2_score(y_test, selected_predictions)
                                        st.metric("R² (Old)", f"{old_r2:.4f}")
                                    with col2:
                                        st.metric("R² (New)", f"{new_r2:.4f}",
                                                 delta=f"{(new_r2-old_r2):.4f}")
                                    with col3:
                                        st.metric("RMSE (Old)", f"{rmse:.4f}")
                                    with col4:
                                        st.metric("RMSE (New)", f"{new_rmse:.4f}",
                                                 delta=f"{(new_rmse-rmse):.4f}")
                                
                                # Store retrained model and predictions
                                st.session_state[f'retrained_{selected_model_name}'] = {
                                    'model': new_model,
                                    'predictions': new_predictions,
                                    'is_retrained': True
                                }
                                
                            except Exception as e:
                                st.error(f"Error retraining model: {str(e)}")
                    
                    with col2:
                        st.info("💡 Adjust parameters and click 'Retrain Model' to compare old vs new performance")
                
                st.markdown("---")
                
                # Detailed visualizations for selected model
                st.markdown(f"<h5 style='color: #667eea;'>Detailed Analysis: {selected_model_name}</h5>", unsafe_allow_html=True)
                
                # Use retrained model/predictions if available
                if f'retrained_{selected_model_name}' in st.session_state and st.session_state[f'retrained_{selected_model_name}']['is_retrained']:
                    display_model = st.session_state[f'retrained_{selected_model_name}']['model']
                    display_predictions = st.session_state[f'retrained_{selected_model_name}']['predictions']
                    st.info("📊 Showing results for the retrained model with custom parameters")
                else:
                    display_model = selected_model_obj
                    display_predictions = selected_predictions
                
                if problem_type == 'classification' and n_classes <= 10:
                    col1, col2 = st.columns([1.5, 1])
                    
                    with col1:
                        st.markdown("<p style='color: #764ba2; font-weight: 600;'>Confusion Matrix</p>", unsafe_allow_html=True)
                        cm = confusion_matrix(y_test, display_predictions)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor('#FAFBFF')
                        ax.set_facecolor('#F5F7FA')
                        
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                                   xticklabels=target_classes, yticklabels=target_classes,
                                   ax=ax, cbar_kws={'label': 'Count'}, linewidths=2, linecolor='white',
                                   annot_kws={'size': 11, 'weight': 'bold', 'color': '#424242'})
                        
                        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', color='#424242')
                        ax.set_ylabel('True Label', fontsize=11, fontweight='bold', color='#424242')
                        ax.set_title(f'Confusion Matrix - {selected_model_name}', fontsize=12, fontweight='bold', color='#667eea')
                        ax.tick_params(axis='both', labelcolor='#424242')
                        
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("<p style='color: #764ba2; font-weight: 600;'>Classification Report</p>", unsafe_allow_html=True)
                        report = classification_report(y_test, display_predictions, target_names=target_classes, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(3), use_container_width=True, height=300)
                    
                    # Model-specific visualizations for classification
                    st.markdown("---")
                    st.markdown("<h5 style='color: #667eea;'>Model-Specific Visualizations</h5>", unsafe_allow_html=True)
                    
                    if selected_model_name == 'Decision Tree':
                        st.markdown("<p style='color: #764ba2; font-weight: 600;'>Decision Tree Structure</p>", unsafe_allow_html=True)
                        plot_decision_tree_viz(display_model, feature_names=selected_features, 
                                             class_names=target_classes, max_depth=3)
                    
                    elif selected_model_name in ['Random Forest', 'Gradient Boosting']:
                        st.markdown("<p style='color: #764ba2; font-weight: 600;'>Feature Importances</p>", unsafe_allow_html=True)
                        if plot_feature_importance(display_model, selected_features):
                            st.success(f"✅ Feature importance plot displayed for {selected_model_name}")
                    
                    elif selected_model_name == 'Support Vector Machine':
                        st.info("ℹ️ SVM is a kernel-based model. Feature importance is not directly available.")
                        st.markdown("<p style='color: #667eea;'>For SVM, examine the decision boundary through prediction patterns.</p>", unsafe_allow_html=True)
                
                elif problem_type == 'regression':
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<p style='color: #764ba2; font-weight: 600;'>Predicted vs Actual</p>", unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor('#FAFBFF')
                        ax.set_facecolor('#F5F7FA')
                        
                        ax.scatter(y_test, display_predictions, alpha=0.6, color='#5DA9E8', edgecolor='white', linewidth=1, s=50)
                        min_val = min(y_test.min(), display_predictions.min())
                        max_val = max(y_test.max(), display_predictions.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                        
                        ax.set_xlabel('Actual Values', fontsize=11, fontweight='bold', color='#424242')
                        ax.set_ylabel('Predicted Values', fontsize=11, fontweight='bold', color='#424242')
                        ax.set_title(f'Predicted vs Actual - {selected_model_name}', fontsize=12, fontweight='bold', color='#667eea')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis='both', labelcolor='#424242')
                        
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("<p style='color: #764ba2; font-weight: 600;'>Residuals Analysis</p>", unsafe_allow_html=True)
                        residuals = y_test - display_predictions
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor('#FAFBFF')
                        ax.set_facecolor('#F5F7FA')
                        
                        ax.hist(residuals, bins=20, color='#5DA9E8', edgecolor='white', linewidth=1.5, alpha=0.7)
                        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                        
                        ax.set_xlabel('Residuals', fontsize=11, fontweight='bold', color='#424242')
                        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold', color='#424242')
                        ax.set_title(f'Residuals Distribution - {selected_model_name}', fontsize=12, fontweight='bold', color='#667eea')
                        ax.legend()
                        ax.grid(True, alpha=0.3, axis='y')
                        ax.tick_params(axis='both', labelcolor='#424242')
                        
                        st.pyplot(fig)
                    
                    # Model-specific visualizations for regression
                    st.markdown("---")
                    st.markdown("<h5 style='color: #667eea;'>Model-Specific Visualizations</h5>", unsafe_allow_html=True)
                    
                    if selected_model_name == 'Decision Tree':
                        st.markdown("<p style='color: #764ba2; font-weight: 600;'>Decision Tree Structure</p>", unsafe_allow_html=True)
                        plot_decision_tree_viz(display_model, feature_names=selected_features, 
                                             class_names=None, max_depth=3)
                    
                    elif selected_model_name in ['Random Forest', 'Gradient Boosting']:
                        st.markdown("<p style='color: #764ba2; font-weight: 600;'>Feature Importances</p>", unsafe_allow_html=True)
                        if plot_feature_importance(display_model, selected_features):
                            st.success(f"✅ Feature importance plot displayed for {selected_model_name}")
                    
                    elif selected_model_name == 'Support Vector Machine':
                        st.info("ℹ️ SVM is a kernel-based model. Feature importance is not directly available.")
                        st.markdown("<p style='color: #667eea;'>For SVM, examine the decision boundary through prediction patterns.</p>", unsafe_allow_html=True)
                
                # Additional model insights
                st.markdown("---")
                st.markdown("<h5 style='color: #667eea;'>Model Information</h5>", unsafe_allow_html=True)
                
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%); 
                                padding: 15px; border-radius: 10px; border-left: 4px solid #5DA9E8;'>
                        <h6 style='color: #1565C0; margin: 0;'>Algorithm</h6>
                        <p style='color: #424242; margin: 5px 0; font-weight: 600;'>{selected_model_name}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with info_col2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #FFF3E0 0%, #FCE4EC 100%); 
                                padding: 15px; border-radius: 10px; border-left: 4px solid #FFB347;'>
                        <h6 style='color: #E65100; margin: 0;'>Training Samples</h6>
                        <p style='color: #424242; margin: 5px 0; font-weight: 600;'>{len(X_train)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with info_col3:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%); 
                                padding: 15px; border-radius: 10px; border-left: 4px solid #66BB6A;'>
                        <h6 style='color: #2E7D32; margin: 0;'>Test Samples</h6>
                        <p style='color: #424242; margin: 5px 0; font-weight: 600;'>{len(X_test)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model hyperparameters (if available)
                if hasattr(display_model, 'get_params'):
                    with st.expander("📋 Model Hyperparameters"):
                        params = display_model.get_params()
                        params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
                        st.dataframe(params_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Tip: Make sure your data is properly formatted and has no issues.")


def main():
    # Initialize UX state
    init_undo_redo_state()
    
    st.sidebar.markdown("""
    <h2 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               background-clip: text; text-align: center; padding: 10px;'>
    📊 EDA Dashboard
    </h2>
    """, unsafe_allow_html=True)
    
    # Undo/Redo Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #667eea;'>↩️ History</h4>", unsafe_allow_html=True)
    undo_col, redo_col = st.sidebar.columns(2)
    with undo_col:
        if st.button("↶ Undo", use_container_width=True):
            undo_result = undo_action()
            if undo_result:
                st.success(f"Undone: {undo_result['action']}")
            else:
                st.info("Nothing to undo")
    with redo_col:
        if st.button("↷ Redo", use_container_width=True):
            redo_result = redo_action()
            if redo_result:
                st.success(f"Redone: {redo_result['action']}")
            else:
                st.info("Nothing to redo")
    
    st.sidebar.markdown(f"<small>Undo: {len(st.session_state.get('undo_stack', []))} | Redo: {len(st.session_state.get('redo_stack', []))}</small>", unsafe_allow_html=True)
    
    # Data upload section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <p style='color: #667eea; text-align: center; font-weight: 600;'>
    Upload a dataset once and then pick the analysis panel.
    </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    # Track upload history
    if uploaded_file:
        add_to_upload_history(uploaded_file.name, uploaded_file.name)
    
    # Show recent files
    history = load_upload_history()
    if history and len(history) > 0:
        with st.sidebar.expander("📁 Recent Files"):
            st.markdown("**Recent uploads (click to note):**")
            for i, item in enumerate(history[:5]):
                st.markdown(f"""
                **{i+1}. {item['filename']}**
                - Last accessed: {item['last_accessed']}
                - Access count: {item['access_count']}
                """)
    
    activity = st.sidebar.radio(
        "Choose view",
        ["EDA(basic)", "Sweetviz", "Plots", "Handle Missing Values", "Preprocess Data", "ML Models"],
    )
    
    # Saved Pipelines section
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #667eea;'>💾 Saved Pipelines</h4>", unsafe_allow_html=True)
    
    pipelines = load_saved_pipelines()
    if pipelines:
        with st.sidebar.expander("📋 My Pipelines"):
            for pipeline_name in list(pipelines.keys()):
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.markdown(f"**{pipeline_name}**")
                    st.caption(f"Created: {pipelines[pipeline_name]['created_at']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{pipeline_name}"):
                        if delete_pipeline(pipeline_name):
                            st.success(f"Deleted: {pipeline_name}")
                            st.rerun()

    # ==================== MAIN CONTENT ====================
    if uploaded_file is None:
        st.markdown("""
        <h1 style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   background-clip: text;'>EDA Dashboard</h1>
        """, unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align: center; color: #764ba2;'>Use the sidebar to upload a CSV file and explore the dataset using multiple analysis panels.</h4>",
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(245, 87, 108, 0.1)); 
                    padding: 30px; border-radius: 15px; border-left: 5px solid #667eea; text-align: center;'>
            <h3 style='color: #667eea; margin-top: 0;'>Start Your Data Analysis Journey</h3>
            <p style='color: #764ba2; font-size: 16px; font-weight: 600;'>
            Start by uploading a CSV file in the left sidebar to begin exploring!
            </p>
            <br/>
            <p style='color: #667eea; font-size: 14px;'>
            💡 <b>Tip:</b> Click "🎓 Start Interactive Tutorial" in the sidebar to learn how to use this dashboard!
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = load_dataset(uploaded_file)

    st.markdown(f"""
    <h2 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               background-clip: text;'>
    Dataset: <span style='color: #667eea;'><b>{uploaded_file.name}</b></span>
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
    elif activity == "ML Models":
        render_models(df)


if __name__ == "__main__":
    main()
