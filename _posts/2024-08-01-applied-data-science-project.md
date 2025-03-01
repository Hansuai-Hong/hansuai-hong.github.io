---
layout: post
author: HanSuai, Hong
title: "Applied Data Science Project Documentation - "
categories: ITD214
---
## Project Background 
The beauty industry is constantly evolving, with shifting consumer preferences, emerging market trends, and increasing competition. To maintain a strong market position, Sephora Products must leverage data-driven strategies to enhance customer experience and drive business growth.

This project focuses on utilizing customer insights, product performance data, and market trends to gain a deeper understanding of customer behavior and industry dynamics. By analyzing purchasing patterns, customer feedback, and engagement metrics, Sephora can make informed decisions that improve product offerings, optimize marketing strategies, and refine pricing structures. A data-driven approach ensures the brand remains responsive to consumer demands while maximizing operational efficiency and profitability.
![image](https://github.com/user-attachments/assets/30cd6596-a645-402e-bc02-ad617bcb0fa5)

## Group Objective
This project is driven by 4 key objectives, each led by a dedicated team member:

1) Sentiment Analysis –
   Analyze customer reviews, feedback, and social media sentiment to understand customer perceptions of Sephora’s products. This insight helps identify areas for improvement and enhance customer satisfaction.

2) Customer Segmentation and Preference Analysis –
   Categorize customers based on demographics, purchasing behavior, and preferences to create targeted marketing strategies and personalized experiences.

3) Develop a Product Recommendation System –
   Build a recommendation engine using machine learning to suggest relevant products based on customer preferences and browsing history, enhancing user engagement and sales.

4) Price Optimization –
   Implement data-driven pricing models that consider demand patterns, competitor pricing, and customer willingness to pay, ensuring competitive yet profitable pricing strategies.

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
two different approaches:
      - Random Forest: A traditional machine learning model that leverages decision trees to classify reviews as positive or negative.
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
   

### 1) Exploratory Data Analysis (EDA)
As part of my role, I identified Product ID, Rating, and Text Review as the key parameters for sentiment analysis. Before proceeding, I conducted a simple Exploratory Data Analysis (EDA) to understand the dataset's structure and distribution on the parameters which identified

Overall Rating Distribution:

<img src="https://github.com/Hansuai-Hong/hansuai-hong.github.io/blob/master/assets/2.png" alt="Description" width="400" height="300">

Positive reivew (rating-4 and rating-5)
negative review (rating-1, rating-2, and rating-3)

    df_reviews['sentiment'] = df_reviews['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')

<img src="https://github.com/Hansuai-Hong/hansuai-hong.github.io/blob/master/assets/3.png" alt="Description" width="400" height="300">

### 2) Data Cleaning
To ensure data accuracy and consistency, I performed the following tasks for cleaning:
- Removing missing values (which less than 1% of total data set)
- Removing duplicate entries to prevent redundancy in the dataset.
- Convert the submission time to datetime format
- Eliminate irrelevant columns that do not contribute to sentiment analysis.
 
      df_reviews1 = df_reviews.dropna(subset=['review_text'])
      df_reviews1['submission_time'] = pd.to_datetime(df_reviews1['submission_time'])
      df_reviews2 = df_reviews1.drop_duplicates()
      df_reviews3 = df_reviews2[['review_text' ,'rating', 'product_id']
       

### 3) Data preprocessing
after data is cleaned, I performed several preprocessing steps, including:

Standardizing text formats by converting all text to lowercase for consistency.
Punctuation and Special Character Removal: Cleaning text by eliminating unnecessary symbols.
Tokenization: Splitting customer reviews into individual words.
Stopword Removal: Removing common words that do not provide meaningful insights.
Lemmatization: Converting words to their root form for consistency.

This step was crucial for optimizing the data for sentiment analysis and predictive modeling using machine learning techniques.

        def clean_text(text):
          # Check if text is NaN or empty, handle it to avoid errors
          if isinstance(text, str):
          # Lowercase the text
          text = text.lower()
        
          # Remove special characters and numbers, keeping spaces
          text = re.sub(r'[^a-z\s]', '', text)
        
          # Tokenize the text into words
          words = nltk.word_tokenize(text)
        
          # Remove stopwords and words with length less than 3
          stop_words = set(stopwords.words('english'))
          words = [word for word in words if word not in stop_words and len(word) >= 3]
        
          # Lemmatize each word in the list
          lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        
          # Reconstruct the cleaned text from lemmatized words
          text = ' '.join(lemmatized_words)
    
        return text

### 4) Filtering
Since performing sentiment analysis on 1 million reviews across 800 products is too generic, I have implemented ipywidgets to allow users to filter results by Product ID. Once a specific product is selected, we can then analyze its reviews in detail.

    # Create an interactive widget
    interactive_widget = widgets.interactive(filter_reviews, product_id=product_input)

    # Display the input widget
    display(interactive_widget)

### 5) Objective I - Understanding Customer Sentiment with Key Insights
We first analyze the text of all reviews to identify the most commonly used words. This helps us determine recurring themes and patterns across the dataset.

    # calculate the number of occurence of each word in the entire list of words
    all_words_frequency = FreqDist(all_words)
    print (all_words_frequency)

    # print 10 most frequently occurring words
    print ("\nTop 50 most frequently occurring words")
    print (all_words_frequency.most_common(50))
This is the most common 50words used in the particular selected product:

<img src="https://github.com/Hansuai-Hong/hansuai-hong.github.io/blob/master/assets/6.png" alt="Description" width="400" height="300">

To gain deeper insights, we categorize reviews into positive and negative sentiments. By isolating the most frequently mentioned words in positive reviews, we can highlight the aspects of Sephora products that customers appreciate the most. Similarly, analyzing negative reviews allows us to pinpoint common complaints or areas where improvements may be needed.



We utilize word clouds to represent the pros and cons of Sephora products based on customer feedback to effectively visualize the findings. Word clouds provide an intuitive way to showcase prominent words, making it easier to recognize key attributes associated with customer satisfaction and dissatisfaction. This approach helps us extract meaningful insights from large volumes of review data, ultimately contributing to a better understanding of customer preferences and areas for product enhancement.




### 6) Objective II - Random Forest: A traditional machine learning model







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
