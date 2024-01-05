# Time-Series Sales Forecasting

## Project Overview
SimpleBuy is a thriving clothing company with a diverse product range aimed at meeting every individual's clothing needs. With both offline and online sales channels, their stock runs out rapidly, which underscores the importance of efficient stock management. This project focuses on forecasting SimpleBuy's sales to aid in the planning of the manufacturing process and raw material procurement.

## Problem Statement
SimpleBuy needs to anticipate sales accurately to maintain their high standard of customer experience. The company has provided two years of sales data and requires sales predictions for the upcoming six months. The forecast will inform their stock management and manufacturing planning.

## Data
The data directory contains training and validation datasets.

```plaintext
TIME-SERIES-PROJECT-SALES-FORECASTING
├── data
│   ├── train
│   └── valid
├── env
├── myenv_salesforecasting
├── notebook
│   └── sales_forecast.ipynb
├── .gitignore
└── readme.md
```

## Approach
The project includes data preprocessing, feature extraction, time series decomposition, and implementation of various forecasting models including Holt's Winters, SARIMA, and machine learning models such as Linear Regression and Random Forest.

## Requirements
The analysis is conducted using Python with the following libraries:

- itertools
- math
- matplotlib
- numpy
- pandas
- scipy
- seaborn
- statistics
- tqdm
- warnings

## Installation
To install the required Python packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
The main analysis can be found in the `sales_forecast.ipynb` Jupyter notebook within the `notebook` directory.

## Conclusion
This project provides a comprehensive approach to sales forecasting, leveraging various statistical and machine learning techniques to deliver accurate predictions that will aid SimpleBuy in effective stock management.