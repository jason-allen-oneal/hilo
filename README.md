# Hilo - BTC Trading ML Model

A machine learning model for predicting BTC price direction using historical candlestick data.

## Installation

```bash
pip install -r requirements.txt
```

## Downloading Historical Data

To improve model accuracy, download more historical data:

```bash
# Download 12 months (default)
python scripts/download_historical_data.py

# Download specific date range
python scripts/download_historical_data.py --start-date 2023-01-01 --end-date 2024-01-01

# Download from Binance instead
python scripts/download_historical_data.py --exchange binance --months 24
```

Then retrain the model:
```bash
python -m lib.model.train_eval
```

## Training the Model

The model training pipeline includes dataset building, hyperparameter tuning, training, and evaluation:

```bash
python -m lib.model.train_eval
```

### Training Options

- `--candles`: Input 1m candle CSV (default: `data/historical_1m.csv`)
- `--dataset`: Intermediate dataset CSV (default: `data/round_dataset.csv`)
- `--model-out`: Output model file (default: `lib/model/model.joblib`)
- `--buffer-size`: Rolling candle buffer size (default: 60)
- `--decision-window-minutes`: Minutes to decide direction after open (default: 2)
- `--test-fraction`: Hold-out fraction for evaluation (default: 0.2)
- `--C-grid`: Comma-separated C values to try (default: "0.1,0.5,1.0,2.0,5.0")
- `--class-weight`: Class weight strategy (choices: "none", "balanced", default: "none")
- `--calibrate`: Enable probability calibration with sigmoid scaling
- `--model-type`: Model type (choices: "logistic", "xgboost", default: "xgboost")

## Data Format

The historical data CSV should have the following columns:
- `open_time`: Unix timestamp in milliseconds
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

## Project Structure

```
hilo/
├── data/                          # Historical data files
│   ├── historical_1m.csv         # 1-minute candlestick data
│   └── round_dataset.csv         # Processed dataset for training
├── lib/                          # Core library code
│   ├── model/                    # ML model code
│   │   ├── train_eval.py        # Training pipeline
│   │   ├── dataset.py           # Dataset building
│   │   └── ...
│   └── ...
├── scripts/                      # Utility scripts
│   └── download_historical_data.py  # Data downloader
└── requirements.txt              # Python dependencies
```
