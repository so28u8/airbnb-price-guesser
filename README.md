# Budapest Airbnb Price Estimator 🇭🇺🏡

A modern, interactive web-based dashboard and machine learning price estimator that analyzes the Budapest Airbnb market data and helps hosts determine the optimal nightly price for their listings.

## 🌟 Features
* **Market Map & Heatmap**: Visualization of average prices and distribution of Budapest accommodations on interactive Plotly maps (density and district bubble views).
* **Statistics & Insights**: Top 10 most expensive districts, price distribution by room type, and general market metrics (median price, average ratings).
* **Price Estimator Model**: A machine learning model based on `HistGradientBoostingRegressor` that estimates the optimal rental price based on the location (district), size (bedrooms, bathrooms), type, and expected ratings.

## 🛠 Tech Stack
* **Language**: Python 3.x
* **Web Framework**: [Streamlit](https://streamlit.io/) (featuring a custom dark design with glassmorphism)
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (HistGradientBoostingRegressor, Pipeline, RandomizedSearchCV), Joblib
* **Visualization**: Plotly Express & Graph Objects

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/so28u8/airbnb-price-guesser.git
cd airbnb-price-guesser
```

### 2. Install dependencies
Create and activate a virtual environment, then install the required packages:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install streamlit pandas numpy scikit-learn plotly joblib
```

### 3. Download the dataset (Optional for running, Required for training)
Since the original `listings.csv` file is too large, it is not included in the repository. If you want to retrain the model:
1. Download the Budapest data from [Inside Airbnb](https://insideairbnb.com/get-the-data/).
2. Name it `listings.csv` and place it in the project root directory.
3. Run the model training script: `python program/model_trainer.py`

*(The repository already contains the trained model in the `program/model/` directory, so the application will run without the dataset!)*

### 4. Start the application
From the project root directory (or the `program` directory), start the Streamlit server:
```bash
streamlit run program/app.py
```
The application will automatically open in your browser at `http://localhost:8501`.

## 📁 Project Structure
* `program/app.py` - The main code for the Streamlit dashboard and UI.
* `program/data_loader.py` - Script responsible for data cleaning and loading.
* `program/model_trainer.py` - Script for training and optimizing the machine learning model.
* `program/model/` - Directory containing the pre-trained model (`price_model.pkl`) and district statistics.
* `spec.pdf` - The original project specification.
