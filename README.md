# Exploratory Data Analysis (EDA) Web App

A powerful and interactive **Streamlit-based EDA application** that allows users to upload datasets and perform quick data analysis, visualization, and preprocessing — all in one place.

---

## Features

### Upload Dataset
- Upload CSV files easily (up to 200MB depending on Streamlit config)
- Instant preview of dataset

### Basic EDA
- View dataset shape (rows × columns)
- List all column names
- Check data types
- Detect missing (null) values
- Summary statistics (mean, std, min, max)
- Select and view specific columns
- Value counts for categorical data

### Advanced EDA Reports
- **Pandas Profiling Report** (auto-generated detailed report)
- **Sweetviz Analysis** (beautiful comparative visual report)

### Visualizations
- Histogram
- Boxplot
- Column-wise analysis

### Data Preprocessing
- Remove null values
- Drop duplicate rows
- Clean dataset preview

---

## Correlation Heatmap
- Visualize relationships between numerical features
- Identify feature dependencies and multicollinearity
- Color-coded matrix for easy understanding

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

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License.

##  Tech Stack

* **Frontend & App Framework:** Streamlit
* **Data Processing:** Pandas
* **Visualization:** Matplotlib, Seaborn
* **EDA Reports:** ydata-profiling, Sweetviz

---

##  Installation

Clone the repository:

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
