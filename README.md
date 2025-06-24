# Diabetes Prediction Project

A machine learning web application for predicting the likelihood of diabetes in patients based on medical attributes. This project uses the PIMA Indians Diabetes dataset and provides a user-friendly interface for predictions.

## Features
- **Data Ingestion:** Loads and splits the dataset for training and testing.
- **Data Transformation:** Preprocesses data using scaling and imputation.
- **Model Training:** Trains multiple models (Logistic Regression, Random Forest, KNN, SVM) and selects the best.
- **Prediction Pipeline:** Loads the trained model and preprocessor to make predictions on new data.
- **Web Interface:** Flask-based web app for user input and displaying predictions.
- **Logging & Exception Handling:** Custom logging and error management for debugging and traceability.

## Project Structure
```
├── app.py                  # Flask app entry point
├── requirements.txt        # Python dependencies
├── setup.py                # Project setup script
├── artifacts/              # Saved models, preprocessors, and datasets
├── logs/                   # Log files
├── notebook/               # Jupyter notebook for EDA and prototyping
├── src/                    # Source code
│   ├── components/         # Data ingestion, transformation, and model training modules
│   ├── pipeline/           # Prediction and training pipelines
│   ├── utils.py            # Utility functions
│   ├── logger.py           # Logging setup
│   └── exception.py        # Custom exceptions
├── templates/              # HTML templates for Flask
```

## Setup Instructions
1. **Clone the repository:**
   ```powershell
   git clone https://github.com/TcheugoueGilbert/diabete_prediction
   cd diabete_prediction
   ```
2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv environment_name

   or

   conda create -p environment_name python==python_version
  

   conda activate envonment_name
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```powershell
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Usage
- Go to the home page and fill in the medical details in the form.
- Submit to get a prediction (Diabetic or Non-Diabetic).

## Data
- The project uses the PIMA Indians Diabetes dataset (`notebook/data/diabetes.csv`).

## Notebooks
- See `notebook/Diabetes_prediction.ipynb` for exploratory data analysis and model prototyping.

## Custom Modules
- **src/components/**: Data ingestion, transformation, and model training logic.
- **src/pipeline/**: Prediction and training pipelines.
- **src/utils.py**: Utility functions for saving/loading models and evaluation.
- **src/logger.py**: Logging configuration.
- **src/exception.py**: Custom exception handling.

## License
This project is licensed under the MIT License.

## Author
- Gilbert Tcheugoue (<djibriltcheugoue@gmail.com>)
- Linkedin (<https://www.linkedin.com/in/gilberttcheugoue>)