# Retail-Customer-Analysis-and-Prediction

## Introduction
This project delves into retail customer behavior, aiming to unlock insights on customer value and segmentation within the retail industry. By leveraging the comprehensive OLIST dataset, this analysis seeks to identify purchasing patterns and predict customer-related attributes using advanced machine learning techniques, including NLP for sentiment analysis, unsupervised learning for customer segmentation, supervised learning for predictive modeling and at last a time series modeling using LSTM

## Objectives
  Pattern Identification: Use machine learning to uncover hidden patterns and trends in customer purchasing behaviors, revealing customer value.
  Predictive Analytics: Employ predictive models to forecast customer attributes, improving business strategies and customer engagement.

## Data Source
The project utilizes the OLIST dataset, encompassing customer, orders, order items, reviews, and product information. After merging and cleaning the datasets, the following columns were used for analysis: order ID, order status, order purchase timestamp, product ID, price, freight value, customer unique ID, customer city, review score, review comment message, product category name (English), delivery time, month, and year.

## Technologies and Libraries Used
This project utilizes Python, with libraries including Pandas for data manipulation, NumPy for numerical computations, Matplotlib and Seaborn for visualization, NLTK and TextBlob for natural language processing, and Scikit-learn, XGBoost, LightGBM, and TensorFlow for machine learning and deep learning models.

## Project Structure and Implementation

The retail-customer-analysis-prediction project is organized into three main directories for streamlined analysis: Src/, Notebooks/, and data/. The src/ directory contains Python scripts for specific tasks: NLP_Sentiment_Analysis.py for analyzing customer reviews, supervised_models.py and unsupervised_models.py for predictive and clustering models, and Time_Series_Analysis.py for sales forecasting using LSTM. These scripts form the analytical backbone of the project. Jupyter notebooks in the notebooks/ directory, including EDA.ipynb for data exploration and Modelling.ipynb for integrating and showcasing the modeling process by using those python files that are in Src, provide an interactive look at the analysis. The data/ directory houses the raw and processed datasets, ready for analysis.

## NLP - Sentiment Analysis
NLP techniques were applied to the review_comment_message field to classify customer sentiments into positive, negative, and neutral categories. This analysis involved text preprocessing, sentiment scoring, and model training to understand customer feedback trends, contributing to a holistic view of customer satisfaction and behavior.

## Unsupervised Learning - Customer Segmentation
Using features such as recency (time since last purchase), frequency (number of purchases), and monetary value (total spent), customers were segmented into four groups: Engaged Regulars, High-Value Newcomers, Occasional Shoppers, and Recent Economical Buyers. This segmentation utilized clustering techniques like Birch and KMeans, offering a nuanced understanding of customer behavior for targeted marketing efforts and KMeans performed better.

## Supervised Learning - Predictive Modeling
Supervised learning models were developed to predict customer spending and other attributes. The models leveraged a mix of features including recency, frequency, monetary value, freight value, delivery time, and review scores. By incorporating the customer segments as additional features, we observed a significant improvement in model performance. Models included Linear Regression as a baseline, followed by more complex models such as Random Forest, LightGBM, XGBoost and Random Forest stands out as the top performer. This phase demonstrated the value of integrating behavioral insights into predictive models.

## LSTM for Time Series Analysis
The time series analysis, conducted using LSTM models, aimed to predict daily sales monetary values. Despite the complexity of forecasting in the unpredictable retail environment, the models provided valuable insights. Model 1 which is the basic LSTM model, despite its simplicity compared to the more complex Model 2 which is the stacked model with additional dropout layers, performed better, although both models faced high RMSE and MAE, highlighting the challenges of using time alone as a predictor for daily sales.

## Future Work
Refining Time Series Forecasting: Further exploration of advanced models and additional variables to enhance the accuracy of daily sales forecasts.
Recommendation Engine Development: Building on the insights from customer segmentation and purchasing behavior, plans include developing a personalized recommendation engine to improve the shopping experience and boost sales.

## Acknowledgments
This project was made possible through the support and guidance of mentors at Lighthouse Lab Bootcamp. Their expertise and encouragement have been instrumental in navigating the complexities of this project, and I am deeply grateful for their mentorship and support.
