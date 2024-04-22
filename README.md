This project focuses on stock market analysis and forecasting using deep learning techniques. I utilizef a range of Python libraries including Pandas, Matplotlib, Numpy, Plotly, and PyTorch to build and implement our models. The objective is to predict stock prices effectively, which could serve as a crucial tool for investors to make informed decisions. Successful trading often requires thorough analysis and prediction strategies, where both fundamental and technical analyses play key roles.

## Implementation

**Data Analysis**   <br>
>In the initial phase of our project, we analyze various attributes of stocks like opening and closing prices. We visualize different aspects such as trends and seasonality in the stock movements.

**Prediction Model**   <br>
>We employ deep learning models to predict future stock prices based on historical data. This includes dividing the data into training and testing sets to evaluate the model's performance.

**Deep Learning Approach**   <br>
>We use models like GRU (Gated Recurrent Unit), which simplifies the process by using fewer parameters than traditional models, while maintaining effectiveness in capturing time dependencies.

**Technology Stack**   <br>
>**Python:** Primary programming language <br>
>**Pandas:** Data manipulation and analysis <br>
>**Matplotlib/Plotly:** Data visualization <br>
>**Numpy:** Numerical operations <br>
>**PyTorch:** Deep learning framework <br>

## Data Description
The analysis includes exploring stock datasets from notorious German companies, such as BMW, Siemens, Allianz and Lufthansa spanning from 2010 to 2020, focusing on attributes like open, close, high, and low prices. 

Techniques used to handle missing values, incorrect data entries, and duplicate data to ensure the quality of the dataset. Methods are employed for transforming stock data into a format suitable for analysis, such as normalization or standardization, to aid in comparative analysis and improve machine learning model performance.

Datasets:

>1.	BMW   <br>
>2.	Siemens   <br>
>3.	Lufthansa   <br>
>4.	Allianz   <br>
	
The distribution of close and open for each is identified, and also the correlation between close and open. After that, attributes [Open, High, Low, Close, volume] of the datasets are visualized for preliminary analysis. Finally, the trend and seasonality in the dataset is identified.

Correlation of "High" values before normalization:

<p align="center">
  <img width="480" height="350" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/HighBeforeNorm.png">
</p>

Correlation of "High" values after normalization:

<p align="center">
  <img width="480" height="350" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/HighAfterNorm.png">
</p>

## Data Analysis
In this step, we delve into the initial analysis performed on the cleaned data, with a focus on:

Exploratory Data Analysis (EDA): Using statistical summaries and visualization techniques to understand the underlying patterns and trends in the stock data.

<p align="center">
  <img width="800" height="400" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/DistributionClose.png">
</p>

Correlation Analysis: Exploring the relationships between different financial indicators such as open and close prices, and their impact on stock predictions. The correlation between Open and Close for example.

<p align="center">
  <img width="800" height="400" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/DistributionOpenClose.png">
</p>

Trend and Seasonality Analysis: Investigating long-term movements in stock prices to identify consistent upward or downward trends, as well as recurring patterns or cycles.

### Allianz

<p align="center">
  <img width="550" height="650" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/AllianzStock.png">
</p>

The trend component extracted from the data highlights a general upward trajectory in the stock value over the decade, suggesting an overall increase in the stock price. There's often a rise early in the year, potentially due to optimism with new fiscal policies or anticipation of strong year-end results reported in this quarter, followed by a dip in Q2, and a mid-year rally is sometimes evident, which could be linked to positive mid-year financial results or seasonal investment trends where investors re-balance portfolios. Often, there's an upward movement towards the end of the year.

### BMW

<p align="center">
  <img width="550" height="650" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/BMWStock.png">
</p>

The BMW stock's close price trend shows a general upward trajectory, with noticeable fluctuations over time. There is a notable increase in average stock prices typically starting from January and peaking around March or April. A slight decline follows this peak, generally around mid-year. Another rise is often seen towards the end of the year, around November and December.

### Lufthansa

<p align="center">
  <img width="550" height="650" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/LufthansaStock.png">
</p>

For Lufthansa, the trend component shows several periods of increase and decline, reflecting the fluctuating nature of the stock price over the decade. Significant declines can be observed around 2011 and another more pronounced one starting in late 2017, which continues until 2020. Spring and Summer usually presents increase, Autumn presents stability or mild increase and the end of the year usually presents dips.

### Siemens

<p align="center">
  <img width="550" height="650" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/SiemensStock.png">
</p>

The trend component indicates a general upward movement over the decade, suggesting long-term growth in the stock value of Siemens. There is a notable increase in stock prices generally starting around March each year. Another peak often occurs towards the end of the year, around November and December. Typically, there's a noticeable dip around June or July. Another dip or period of high variability is often seen in August, possibly due to lower trading volumes during the summer holidays in many regions, which can lead to increased price volatility.

## Deep Learning Approach
Gated Recurrent Units (GRUs) was chosen over other recurrent neural network models for offering a simplified architecture that requires fewer parameters than LSTMs, making them faster to train without a significant trade-off in performance. This reduction in complexity and training time is particularly beneficial for handling the volatile nature of stock price data.

The GRU architecture is designed to better capture time-dependent patterns in sequential data like stock prices. Each GRU cell consists of two gates: a reset gate and an update gate. 

<p align="center">
  <img width="515" height="373" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/GRUModel.webp">
</p>

The training of the GRU model involves initialization, loss function (Mean Squared Error - MSE) to quantify the difference between predicted stock prices and actual values, optimization algorithm like Adam, which adjusts model weights iteratively based on training data to minimize the loss function and training loop, to iterate through epochs where the model learns from the training data in batches.

We plot stock prices over time to illustrate trends and patterns, comparing actual stock prices against predictions from the model to visually assess the model performance.

### Allianz

<p align="center">
  <img width="990" height="400" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/AllianzResult.png">
</p>

<p align="center">
  <img width="980" height="300" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/AllianzPrediction.png">
</p>

### BMW

<p align="center">
  <img width="990" height="400" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/BMWResult.png">
</p>

<p align="center">
  <img width="980" height="300" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/BMWPrediction.png">
</p>

### Lufthansa

<p align="center">
  <img width="990" height="400" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/LufthansaResult.png">
</p>

<p align="center">
  <img width="980" height="300" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/LufthansaPrediction.png">
</p>

### Siemens

<p align="center">
  <img width="990" height="400" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/SiemensResult.png">
</p>

<p align="center">
  <img width="980" height="300" src="https://github.com/gomeslelino/Stock-Market-Forecasting/blob/main/Pictures/SiemensPrediction.png">
</p>

## Conclusion
The model was useful for assessing the behaviour of the stocks of these four companies, for every single one of them, the prediction was very accurate when compared to the real behaviour of the stock. The predictions works best if there is a pattern of cyclical behavior for the stocks.
