---
layout: post
author: HanSuai, Hong
title: "Applied Data Science Project Documentation - "
categories: ITD214
---
## Project Background 
The beauty industry is constantly evolving, with shifting consumer preferences, emerging market trends, and increasing competition. To maintain a strong market position, Sephora Products must leverage data-driven strategies to enhance customer experience and drive business growth.

This project focuses on utilizing customer insights, product performance data, and market trends to gain a deeper understanding of customer behavior and industry dynamics. By analyzing purchasing patterns, customer feedback, and engagement metrics, Sephora can make informed decisions that improve product offerings, optimize marketing strategies, and refine pricing structures. A data-driven approach ensures the brand remains responsive to consumer demands while maximizing operational efficiency and profitability.

## Group Objective
This project is driven by 4 key objectives, each led by a dedicated team member:

1) Sentiment Analysis – Analyze customer reviews, feedback, and social media sentiment to understand customer perceptions of Sephora’s products. This insight helps identify areas for improvement and enhance customer satisfaction.

2) Customer Segmentation and Preference Analysis – Categorize customers based on demographics, purchasing behavior, and preferences to create targeted marketing strategies and personalized experiences.

3) Develop a Product Recommendation System – Build a recommendation engine using machine learning to suggest relevant products based on customer preferences and browsing history, enhancing user engagement and sales.

4) Price Optimization – Implement data-driven pricing models that consider demand patterns, competitor pricing, and customer willingness to pay, ensuring competitive yet profitable pricing strategies.

By integrating these analytical approaches, Sephora Products can improve personalization, strengthen customer loyalty, and increase business profitability.


## Personal Objective
<img src="https://github.com/Hansuai-Hong/hansuai-hong.github.io/blob/master/assets/1.png" alt="Description" width="400" height="300">
Objective

The sentiment analysis in this project aims to extract meaningful insights from customer reviews and develop a predictive model to classify sentiments effectively. This is achieved through two key objectives:

1) Understanding Customer Sentiment
Analyze review text to identify the most common words used across all reviews.
Determine the most frequently mentioned words in positive and negative reviews separately.
Use word clouds to visually represent the pros and cons of Sephora products based on customer feedback.

2) Building a Sentiment Prediction Model
Develop a machine learning model to predict sentiment based on customer reviews.
     - two different approaches:
Random Forest: A traditional machine learning model that leverages decision trees to classify reviews as positive or negative.
     - Recurrent Neural Network (RNN): A deep learning model designed to capture contextual meaning from text data for more accurate sentiment prediction.

These objectives aim to provide Sephora with valuable insights into customer perceptions while enabling automated sentiment classification for future reviews.


## Work Accomplished
### 1) Data Collection (Grouop effort)
As a team, we conducted thorough research to identify a suitable dataset for our analysis. After evaluating multiple sources, we selected a dataset from Kaggle that provides comprehensive insights relevant to our project objectives. 

This dataset includes two key components: 
Info Data - which contains general product details
Review Data - which consists of five sheets of customer reviews and associated information.

After obtaining the dataset, we performed an initial Exploratory Data Analysis (EDA) to gain a better understanding of its structure and quality. This process involved:
- Reviewing data distribution.
- Identifying inconsistencies, missing values, and duplicate entries.
- Detecting outliers.
- Recognizing key patterns to guide deeper analysis.

This preliminary analysis helped ensure that the dataset was well-prepared for subsequent sentiment analysis and predictive modeling tasks.
   

### 1) Data Cleaning
As part of my role, I shortlisted 3 key paramters - Product ID, Rating, and Text review as my primary parameters. 
Before start, simple EDA performed to understand my data set with key parameters. 
overall rating:

<img src="https://github.com/Hansuai-Hong/hansuai-hong.github.io/blob/master/assets/2.png" alt="Description" width="400" height="300">

Positive reivew (rating4 and rating 5) vs negative review (rating1, rating2, and rating3)

df_reviews['sentiment'] = df_reviews['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')

<img src="https://github.com/Hansuai-Hong/hansuai-hong.github.io/blob/master/assets/3.png" alt="Description" width="400" height="300">

Then i start the prework by conducted a thorough data cleaning process to ensure the dataset was accurate and ready for analysis. The key tasks completed include:

- Removing duplicate entries to prevent redundancy in the dataset.
- Handling missing values by either imputing them or removing incomplete records.



Standardizing text formats by converting all text to lowercase for consistency.
Eliminating irrelevant columns that do not contribute to sentiment analysis.
This process improved data quality and prepared the dataset for further preprocessing and modeling.






### Data Preparation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

### Modelling
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

### Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Recommendation and Analysis
Explain the analysis and recommendations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
