# 🚀 Exploratory Data Analysis (EDA) Web App

A powerful and interactive **Streamlit-based EDA application** that allows users to upload datasets, perform comprehensive data analysis, preprocessing, feature engineering, and machine learning model training — all in one intuitive interface.

---

## ✨ Key Features

### 📊 **Data Upload & Exploration**
- **Easy CSV Upload** - Upload datasets up to 200MB
- **Instant Preview** - View first rows of your dataset
- **Upload History** - Track previously uploaded files with timestamps and access counts
- **Data Summary** - Quick statistics and data type detection

### 🔍 **Exploratory Data Analysis (EDA)**

#### Basic EDA
- View dataset shape (rows × columns)
- List all column names with data types
- Detect missing (null) values
- Comprehensive summary statistics (mean, std, min, max, quartiles)
- Select and view specific columns
- Value counts for categorical data

#### Advanced Reports
- **Pandas Profiling** - Auto-generated detailed statistical report
- **Sweetviz Analysis** - Beautiful interactive comparative visual report
- **Column-wise Analysis** - Detailed exploration of individual columns

#### Visualizations
- Histograms for distribution analysis
- Boxplots for outlier detection
- Correlation Heatmap for feature relationships
- Multi-column analysis with customizable plots

### 🔧 **Data Preprocessing & Cleaning** (5 Comprehensive Steps)

#### Step 1: Handle Missing Values
- Multiple strategies: Drop, Mean/Median fill, Mode fill, Forward/Backward fill, Interpolation
- Visual preview of impact
- Missing value pattern visualization

#### Step 2: Remove Duplicates
- Identify and remove duplicate rows
- Smart filtering with subset columns option
- Before/after comparison

#### Step 3: Handle Outliers
- **Multiple Detection Methods**: IQR, Z-score, Isolation Forest, Local Outlier Factor
- Visual visualization of outliers
- Multiple handling strategies: Remove, Cap, Transform
- Before/after comparison

#### Step 4: Feature Scaling
- Multiple scaling methods: Standard Scaling, Min-Max, Robust Scaling
- Selective column scaling
- Preserves original data type information

#### Step 5: **📈 Feature Selection with Insights**
- **Intelligent Feature Ranking** - Automatic importance scoring for all features
- **Visualization** - Bar charts showing feature importance with gradient colors
- **Smart Recommendations** - Top 3 recommended features with explanations
- **Correlation Analysis** - Feature-to-target correlation visualization
- **SelectKBest Algorithm** - Choose top K features for model training
- **Problem Type Detection** - Auto-identifies Classification vs Regression

### 🤖 **Machine Learning Models**

#### Dataset Insights Before Training
- Selected features overview with statistics
- Target column analysis and distribution
- Expected results preview
- Class imbalance detection (for classification)
- Data quality metrics and recommendations

#### **📊 Advanced Feature Importance Analysis**
- **Feature Importance Ranking Table** - All features ranked by importance scores
- **Importance Visualization** - Color-coded bar chart showing feature impact
- **Correlation with Target** - Positive/Negative correlation analysis
- **Smart Recommendations** - Top features highlighted with explanations
- **Problem Type Guidance** - Classification or Regression specific insights

#### Model Training & Evaluation
- **Classification Models** (7 models):
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Naive Bayes

- **Regression Models** (6 models):
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - KNN Regressor
  - Support Vector Regressor (SVR)

#### Comprehensive Model Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression Metrics**: R² Score, Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Model Comparison** - Side-by-side performance comparison
- **Train/Test Split** - Configurable data splitting ratio
- **Detailed Reports** - Classification reports and confusion matrices

### 💾 **Advanced Features**

#### Saved Pipelines
- Save preprocessing pipelines for reuse
- Load previously saved pipelines
- Export pipeline configurations
- Streamlined workflow for multiple datasets

#### Download Options
- Download preprocessed datasets in CSV format
- Export models as pickle files
- Save analysis reports
- Generate detailed model evaluation reports

#### Theme & UI
- Modern, gradient-based dark theme
- Responsive design
- Color-coded insights and recommendations
- Animated transitions for better UX
- Intuitive sidebar navigation

### 📚 **Guided Workflows**
- Step-by-step data preprocessing guide
- ML model selection recommendations
- Best practices for feature engineering
- Data quality assessment and improvement suggestions

---

## 🎯 What's New (Latest Updates)

### Version 2.0 - Feature Intelligence & ML Integration
- ✨ **Feature Importance Insights** in Data Preprocessing Step 5
  - Automatic feature scoring and ranking
  - Visual importance charts with gradient coloring
  - Smart recommendations for feature selection
  - Correlation analysis with target variable

- ✨ **ML Models Section Enhancements**
  - Advanced feature importance analysis before training
  - Data quality assessments and recommendations
  - Problem type detection and guidance
  - Top features highlighting for better accuracy

- 🎨 **UI/UX Improvements**
  - Modern gradient-based dark theme
  - Animated transitions and interactions
  - Color-coded insights and alerts
  - Improved sidebar navigation

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kkk0070/Data-preprocessing-and-EDA-streamlit-.git
   cd Data-preprocessing-and-EDA-streamlit-
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Run the Streamlit app:**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Step-by-Step Workflow

1. **Upload Dataset** → Upload a CSV file
2. **Explore Data** → Choose from EDA(basic), Sweetviz, or Plots
3. **Analyze Missing Values** → Visualize and handle missing data
4. **Preprocess Data** → Apply cleaning and transformation steps (5 steps including feature selection insights)
5. **Train ML Models** → Select features, view importance analysis, and train models
6. **Evaluate Results** → Compare model performance and export results

---

## 📋 Requirements

### Core Dependencies
- **Python** 3.8+
- **Streamlit** 1.20+
- **Pandas** 1.3+
- **NumPy** 1.20+
- **Scikit-learn** 0.24+
- **Matplotlib** 3.3+
- **Seaborn** 0.11+

### EDA & Analysis
- **ydata-profiling** 3.0+
- **Sweetviz** 2.0+

### ML & Preprocessing
- **scikit-learn** (included above)
- **joblib** 1.0+

See `requirements.txt` for complete list and exact versions.

---

## 🏗️ Technical Architecture

### Application Structure
```
Data-preprocessing-and-EDA-streamlit-/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
└── images/                   # Assets and screenshots
```

### Key Components

| Component | Purpose | Technologies |
|-----------|---------|---------------|
| **Frontend** | Interactive UI & visualizations | Streamlit, HTML/CSS |
| **Data Processing** | ETL & transformation | Pandas, NumPy |
| **Visualization** | Charts & heatmaps | Matplotlib, Seaborn |
| **Machine Learning** | Model training & evaluation | Scikit-learn |
| **EDA Reports** | Advanced analysis | ydata-profiling, Sweetviz |

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML & Models** | Scikit-learn |
| **Advanced Analytics** | ydata-profiling, Sweetviz |
| **Language** | Python 3.8+ |

---

## 🎓 Feature Highlights Explained

### Feature Importance Insights
**Why it matters?**
- Helps identify which features are most predictive
- Reduces model complexity by selecting relevant features
- Improves model accuracy and reduces overfitting
- Saves computational resources by eliminating noise

**How to use?**
1. Go to "Preprocess Data" → "Step 5: Feature Selection"
2. Check "Perform feature selection"
3. Select your target column
4. Review the feature importance scores and recommendations
5. Use insights to make informed feature selection decisions

### ML Models Section
**Workflow:**
1. Select target column
2. Choose predictor features
3. Review feature importance before training
4. Train multiple models simultaneously
5. Compare performance metrics
6. Export best performing model

**Supported Problems:**
- **Classification**: Binary and multi-class problems
- **Regression**: Continuous value prediction

---

## 📊 Example Usage

### Scenario: Predicting House Prices

1. **Upload** a housing dataset (CSV)
2. **Explore** the data with basic EDA
3. **Preprocess**:
   - Handle missing price values
   - Remove duplicates
   - Handle outliers in features
   - Scale numerical features
   - **View feature importance** to select best predictive features
4. **Train Models**:
   - **Review feature importance insights** in ML section
   - Select regression models
   - Compare model performance
5. **Export** the best model

### Example: Customer Classification

1. **Upload** customer dataset
2. **Run** Sweetviz analysis for deep insights
3. **Clean Data**: Remove duplicates, handle missing values
4. **Feature Engineering**: Scale features, select important ones
5. **Train Classification Models**:
   - **Review feature importance** for each class
   - Train 7 different classifiers
   - Evaluate confusion matrices
6. **Deploy** the best classifier

---

## 🔒 Data Privacy & Security

- ✅ All processing happens locally
- ✅ No data is sent to external servers
- ✅ No data storage or logging
- ✅ Safe for sensitive datasets

---

## 🐛 Troubleshooting

### Issue: App won't start
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade

# Clear cache and restart
streamlit run app.py --logger.level=debug
```

### Issue: Out of memory with large files
- Use smaller dataset samples
- Close other applications
- Upgrade system RAM

### Issue: Missing values not showing
- Ensure CSV has proper formatting
- Check for different null representations (NA, N/A, blank, etc.)

---

## 📈 Performance Considerations

- **File Size**: Works efficiently with datasets up to 500MB
- **Processing Time**: Depends on dataset size and selected features
- **Memory**: Requires 4GB+ RAM for large datasets
- **Browser**: Works best on modern browsers (Chrome, Firefox, Safari, Edge)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## 👨‍💻 Author

**Kavin K**
- GitHub: [@kkk0070](https://github.com/kkk0070)
- Repository: [Data-preprocessing-and-EDA-streamlit-](https://github.com/kkk0070/Data-preprocessing-and-EDA-streamlit-)

```bash
git clone https://github.com/your-username/eda-streamlit-app.git
cd eda-streamlit-app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas matplotlib seaborn ydata-profiling sweetviz
```

---

##  Run the App

```bash
streamlit run app.py
```

---

##  Project Structure

```
eda-streamlit-app/
│
├── app.py                # Main Streamlit app
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── report.html          # Sweetviz report (generated)
```

---

##  Screenshots

* Sidebar with multiple EDA options
* Dataset preview and summary
* Interactive reports (Pandas Profiling & Sweetviz)
* Correlation heatmap visualization

---

##  Use Cases

* Quick dataset understanding
* Data cleaning before ML
* Feature selection
* Academic projects
* Business data insights

---

##  Limitations

* Works best with CSV files
* Correlation works only with numerical data
* Large datasets may take time for profiling reports

---

##  Future Improvements

* Add Plotly interactive charts
* Drag-and-drop upload UI
* Download processed dataset
* Machine Learning model integration
* Dark/Light theme toggle

---

##  Contributing

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

---

##  License

This project is open-source and available under the **MIT License**.

---

##  Author

**Kavin**

* Passionate about Data Science & Full Stack Development
* Skilled in Python, ML, and Data Analysis

---

## Support

If you like this project, don’t forget to  the repository!
