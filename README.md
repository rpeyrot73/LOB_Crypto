# LOB_Crypto

LOB_Crypto is a Python-based framework designed for high-frequency financial data processing, model training, and backtesting in cryptocurrency markets. It uses Limit Order Book (LOB) data to predict mid-price movements and optimize trading strategies. 
This project is based on the research conducted as part of the study: [*Deep Limit Order Book Forecasting Applied to Crypto Currencies*](research_paper/Deep_Limit_Order_Book_Forecasting_Applied_to_Crypto_Currencies.pdf) and draws significant inspiration from [**LOBFrame**](https://github.com/FinancialComputingUCL/LOBFrame) (Briola et al.), incorporating ideas and methodologies from their research [*Deep Limit Order Book Forecasting*](https://arxiv.org/abs/2403.09267).
This project integrates advanced deep learning techniques such as CNNs, LSTMs, and Transformers to build and evaluate models for historical data.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Citation](#citation)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Acknowledgments](#acknowledgments)
9. [License](#license)

---

## Introduction

LOB_Crypto applies state-of-the-art deep learning models to predict cryptocurrency price movements using Limit Order Book (LOB) data. The project examines how varying liquidity levels and DL architectures affect forecasting accuracy, leveraging a robust pipeline for:
- Data preprocessing.
- Model training, validation and testing.
- Profit and loss (PnL) analysis through backtesting in the test set.

---

## Pre-requisites

Before running the project, ensure the following steps are completed:

1. **Install Required Packages**:
   - Install the Python dependencies specified in the `requirements.txt` file:
   - 
     ```bash
     pip install -r requirements.txt
     ```

2. **Configure the Project**:
   - Adapt the `config.json` file to suit your requirements. Update paths, table names, and other parameters as needed. For example:
   - 
     ```json
     {
         "table": "book",
         "exchanges": "BINANCE",
         "path_sqlite": "database/",
         "path_save_results": "results/",
         "path_save_plots": "plots/"
     }
     ```

---

## Data

This project requires high-frequency financial data fetched using the Crypto Lake API. Follow the steps below to set up the data fetching process:

1. **Create an Account**:
   - Register for an account at [Crypto Lake](https://crypto-lake.com/).

2. **Obtain API Credentials**:
   - You will receive your AWS Access Key ID and Secret Access Key through Crypto Lake.

3. **Store Your Credentials Securely**:
   - Create a `crypto_lake_aws_credentials.txt` file with the following format:
     
     ```
     AWS_ACCESS_KEY_ID=your_access_key_id
     AWS_SECRET_ACCESS_KEY=your_secret_access_key
     region_name=eu-west-1
     ```

4. **Run the Data Fetching Script**:
   - Use the provided Python script (`download_raw_data.py`) to fetch the raw data and fill the sqlite database. This script supports command-line arguments for flexibility.

5. **Database Setup**:
   - The script automatically creates an SQLite database in the specified directory (`path_sqlite`) defined in the config.json file and populates it with cleaned and preprocessed data.


---

## Features

- **Data Pipeline**: Fetch and preprocess large-scale LOB data from a SQLite database.
  - Example command to fetch and preprocess data:
    ```bash
    python main.py --pipeline data --config config.json
    ```

- **Training Pipeline**: Train CNN, LSTM, and Transformer models with PyTorch Lightning.
  - Example command to train models:
    ```bash
    python main.py --pipeline training --training --config config.json
    ```

- **Backtesting Pipeline**: Simulate trading strategies using model predictions and evaluate PnL metrics.
  - Example command to backtest with pre-trained models:
    ```bash
    python main.py --pipeline backtest --run_backtest --config config.json
    ```

- **Customizable Configurations**: Modify paths, model parameters, and trading settings via JSON files.
  - **Example Configuration (`config.json`)**:
    ```json
    {
        "table": "lob_data",
        "exchanges": "BINANCE",
        "path_sqlite": "database/",
        "path_save_results": "results/",
        "path_save_plots": "plots/"
    }
    ```

---

## Citation

If you use this project or build upon it, please cite the associated research paper:

[*Deep Limit Order Book Forecasting Applied to Crypto Currencies*](research_paper/Deep_Limit_Order_Book_Forecasting_Applied_to_Crypto_Currencies.pdf)


```bibtex
@article{peyrota2024deepLOBcrypto,
  title={Deep Limit Order Book Forecasting Applied to Crypto Currencies},
  author={Peyrot, Remi},
  year={2024}
}
```

---

## Project Structure

```
LOB_Crypto/
├── .gitignore                          # Specifies files and folders ignored by version control
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── config.json                         # Configuration file for the project
├── crypto_lake_aws_credentials.txt     # AWS credentials to fetch data via Crypto Lake API (excluded from Git)
├── download_raw_data.py                # Script to download raw data from the Crypto Lake API
├── data_pipeline.py                    # Data preprocessing pipeline
├── training_pipeline.py                # Model training, validation and testing pipeline
├── backtest.py                         # Backtesting and PnL analysis pipeline
├── models_pt.py                        # Deep learning model definitions (e.g., DeepLOB, Transformers)
├── utils_pipeline.py                   # Utility functions for data preprocessing and management
├── main.py                             # Entry point for pipeline execution
├── research_paper/                     # Research paper associated to this project (e.g., PDF)
├── data_preprocessed/                  # Stores preprocessed data (excluded from Git, created during runtime)
└── plots/                              # Stores generated plots (excluded from Git, created during runtime)
```







