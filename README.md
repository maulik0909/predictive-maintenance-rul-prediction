# Predictive Maintenance & Remaining Useful Life (RUL) Prediction

An end-to-end Machine Learning project that predicts the Remaining Useful Life (RUL) of aircraft engines using time-series sensor data from the NASA C-MAPSS FD001 dataset.

The project combines an LSTM deep learning model, a FastAPI prediction service, a Streamlit user interface, and Docker Compose for containerized deployment.

## Project Overview

Aircraft engines generate sensor readings throughout their operation. This project uses historical sensor sequences to estimate how many operational cycles remain before an engine reaches its end of useful life.

### Key Features

- Time-series preprocessing and feature preparation
- LSTM-based Remaining Useful Life prediction
- Saved model and feature scaler for inference
- REST API built with FastAPI
- Interactive dashboard built with Streamlit
- Dockerized backend and frontend
- API evaluation against the FD001 test set

## Technology Stack

- Python
- TensorFlow / Keras
- Pandas and NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Docker and Docker Compose
- Jupyter Notebook

## Model Performance

Evaluation on the NASA C-MAPSS FD001 test set:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 16.95 cycles |
| Root Mean Squared Error (RMSE) | 23.67 cycles |

MAE represents the average absolute difference between predicted and actual RUL. RMSE penalizes larger prediction errors more heavily.

These results are specific to the current test evaluation and should not be interpreted as a guarantee of performance on other engine populations.

## Project Structure

```text
predictive-maintenance/
├── app/
│   └── main.py
├── data/
│   └── raw/                 # Download separately; excluded from Git
├── models/
│   ├── rul_lstm_50.keras
│   └── rul_scaler.pkl
├── notebooks/
├── Dockerfile
├── docker-compose.yml
├── frontend.py
├── evaluate_api.py
├── test_api.py
├── api_evaluation_results.csv
├── requirements.txt
└── README.md
```

## Run with Docker

### Prerequisites

- Docker Desktop installed and running
- The NASA C-MAPSS FD001 dataset downloaded separately
- The required model and scaler files present in `models/`

### Start the application

From the project root, run:

```bash
docker compose up --build
```

Open the following URLs:

- **Streamlit dashboard:** http://localhost:8501
- **FastAPI documentation:** http://localhost:8000/docs
- **API health endpoint:** http://localhost:8000/health

To stop the containers, press `Ctrl+C` and then run:

```bash
docker compose down
```

## Dataset

The project uses the NASA C-MAPSS FD001 turbofan engine degradation dataset.

Download it from the official NASA Prognostics Center of Excellence data repository:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Place the required FD001 files in `data/raw/` using these filenames:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

The raw dataset is excluded from this Git repository. Review the dataset's terms before redistributing it.

## Model Details

The selected model is an LSTM network trained on sequences of 50 engine cycles and 24 input features. The saved scaler is used to standardize sensor features consistently during inference.

The model predicts RUL from a sequence of observed sensor readings. Predictions depend on the input preprocessing and the compatibility of the saved model and scaler.

## Future Improvements

- Add automated tests and continuous integration
- Improve experiment tracking and reproducibility
- Add model monitoring and input validation
- Explore alternative time-series architectures
- Deploy the containerized application to a cloud platform

## Author

Developed as a Machine Learning engineering portfolio project, covering model training, inference API development, evaluation, and containerized deployment.