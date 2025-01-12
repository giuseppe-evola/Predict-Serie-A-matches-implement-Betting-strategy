# Football Results prediction & Betting strategy 

## Objective
Can we apply a data-driven approach to predict the outcomes of football matches and create a profitable betting strategy based solely on data instead of intuition? This project aims to explore this possibility by focusing exclusively on the Italian football league (**Serie A**) and using data that is readily available and easily accessible online.

## Technique Used
There are various approaches to predicting football match outcomes based on historical data from individual teams. Generally, these methods leverage specific averages of statistics over the last **n matches**. In this project, we use **Exponential Moving Averages (EMA)** with a span of **10**. This technique allows us to capture the progressive or regressive trends of each team to improve predictive accuracy.

## Data Source
https://www.football-data.co.uk
This website provides commonly used football statistics, which are freely available and downloadable in `.csv` format.

## Data Used
We utilized data from **Serie A** covering seasons from **2016-2017** to **2024-2025** (up to the 3rd matchday in November). The data includes basic match-level statistics such as **goals, shots (home and away teams), etc.** The raw data used can be found in the folder `Data`.

## Project Structure
The project is divided into four main sections:

### 1. Data Preparation
*Files:* `Data_preparation.ipynb` & `Data_preparation_2.ipynb`
* **Data Download:**
   * The script `DataScraper.py` (written in Python and utilizing **BeautifulSoup**) performs web scraping to download raw data from the aforementioned website.
* **Data Cleaning and Manipulation:**
   * In `Data_preparation.ipynb`, we process the raw data to create the `GENERAL_STATS` dataset, which contains key general statistics for each match.
   * In `Data_preparation_2.ipynb`, we perform feature engineering to generate team-specific data using **Exponential Moving Averages (EMA)**, resulting in the `EMA_data` dataset.

### 2. Data Analysis
*File:* `Data_analysis.ipynb`
* This section involves **in-depth analysis** of the datasets to uncover fundamental relationships between variables.
* While these insights are not directly used for outcome predictions, we prioritize feeding the models comprehensive datasets to maximize the information available during training.

### 3. Results Prediction
*File:* `Prediction_models.ipynb`
* Various **machine learning models** are applied in this section.
* Models are trained and saved in the `ML_models` folder for use in subsequent steps.
* The implementation is done via `MachineLearningModels.py`, which follows a structured pipeline for applying and evaluating these models.

### 4. Betting Strategy
*File:* `Betting_strategy.ipynb`
* In this section, we evaluate the performance of the prediction models to identify which one produces the most accurate forecasts.
* Using a **basic betting strategy**, we analyze profitability and demonstrate that, with a data-driven approach, it is possible to outperform bookmakers using relatively straightforward mathematical and statistical tools.

## Notes
* Each phase of the project is discussed in detail in its corresponding file. As a result, there is no single comprehensive report but rather a dedicated notebook for each section.
* We recommend following the project in the **order outlined above**.

## Disclaimer
This project is intended for **educational purposes only**. The authors take no responsibility for any betting decisions made based on this work.
