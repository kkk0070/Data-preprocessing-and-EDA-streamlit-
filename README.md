# Exploratory Data Analysis (EDA) Web App

A powerful and interactive **Streamlit-based EDA application** that allows users to upload datasets and perform quick data analysis, visualization, and preprocessing — all in one place.

![App Overview](images/app_overview.png)

---

## Features

### Upload Dataset
- Upload CSV files easily (up to 200MB depending on Streamlit config)
- Instant preview of dataset

![Upload Interface](images/upload_interface.png)

### Basic EDA
- View dataset shape (rows × columns)
- List all column names
- Check data types
- Detect missing (null) values
- Summary statistics (mean, std, min, max)
- Select and view specific columns
- Value counts for categorical data

![Basic EDA](images/basic_eda.png)

### Advanced EDA Reports
- **Pandas Profiling Report** (auto-generated detailed report)
- **Sweetviz Analysis** (beautiful comparative visual report)

![Pandas Profiling Report](images/pandas_profiling.png)

### Visualizations
- Histogram
- Boxplot
- Column-wise analysis

![Visualizations](images/visualizations.png)

### Data Preprocessing
- Remove null values
- Drop duplicate rows
- Clean dataset preview

---

## Correlation Heatmap
- Visualize relationships between numerical features
- Identify feature dependencies and multicollinearity
- Color-coded matrix for easy understanding

![Correlation Heatmap](images/correlation_heatmap.png)

---

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Data-preprocessing-and-EDA-streamlit-
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

Open your browser and navigate to the provided URL to start using the EDA web app.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pandas Profiling
- Sweetviz

See `requirements.txt` for full list of dependencies.

## Tech Stack

* **Frontend & App Framework:** Streamlit
* **Data Processing:** Pandas
* **Visualization:** Matplotlib, Seaborn
* **EDA Reports:** ydata-profiling, Sweetviz

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License.

If you like this project, don’t forget to  the repository!
