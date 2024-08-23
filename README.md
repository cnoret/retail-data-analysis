# Weekly Sales Prediction Project

## Overview

This project is designed to analyze historical sales data from a large retail chain and predict weekly sales using machine learning models. The project involves data preprocessing, exploratory data analysis (EDA), and the development of predictive models to forecast sales. The project is implemented in Python and presented using a Streamlit web application.

## Features

- **Data Exploration:** Analyze the historical sales data, visualize trends, distributions, and relationships between different variables.
- **Correlation Analysis:** Compute and visualize the correlation matrix to understand the relationships between different features.
- **Sales Trend Analysis:** Explore the sales trends over time to identify patterns and seasonal effects.
- **Predictive Modeling:** Use machine learning models (e.g., Linear Regression, Random Forest Regressor) to predict weekly sales.
- **Interactive Predictions:** Allow users to input data and generate predictions for weekly sales using the trained models.

## Technologies Used

- **Python 🐍**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-Learn
  - Streamlit
- **Development Tools**
  - Jupyter Notebook
  - Visual Studio Code
  - Git
  - GitHub

## Project Structure

- `data/` - Folder containing the dataset files.
- `images/` - Folder containing images used in the project.
- `app.py` - The main Streamlit application file.
- `README.md` - The file you are currently reading.
- `requirements.txt` - Python dependencies required to run the project.

## Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed on your machine. You'll also need to install the dependencies listed in the `requirements.txt` file.

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/weekly-sales-prediction.git
    cd weekly-sales-prediction
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. **Navigate to the project directory:**

    ```bash
    cd weekly-sales-prediction
    ```

2. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

3. **Access the application in your browser:**
   - The application should automatically open in your default web browser. If not, navigate to `http://localhost:8501/` in your browser.

### Project Walkthrough

1. **Data Exploration:**
   - Start by exploring the datasets used in the project. The app will display data overviews, summaries, and visualizations to help understand the data.

2. **Correlation Matrix:**
   - Visualize the correlation matrix to understand the relationships between different features in the data.

3. **Sales Trend Analysis:**
   - Analyze the sales trends over time to identify significant patterns and peaks.

4. **Modeling:**
   - Train machine learning models to predict weekly sales based on historical data. The app allows you to choose between different models and evaluate their performance.

5. **Make Predictions:**
   - Use the app to input new data and generate predictions for weekly sales using the trained models.

## License

This project is open-source and available under the [MIT License](LICENSE).
