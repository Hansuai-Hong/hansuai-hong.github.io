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
<img src="https://hansuai-hong.github.io/assets/1.png" alt="Description" width="400" height="300">
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
To perform sentiment analysis, I had identified Product ID, Rating, and Text Review as my key parameters to analysis. Before proceeding, I conducted a simple Exploratory Data Analysis (EDA) to understand the dataset's structure and distribution on the parameters which identified

Overall Rating Distribution:

<img src="https://hansuai-hong.github.io/assets/2.png" alt="Description" width="400" height="300">

Positive reivew (rating-4 and rating-5)
negative review (rating-1, rating-2, and rating-3)

      df_reviews['sentiment'] = df_reviews['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')

<img src="https://hansuai-hong.github.io/assets/3.png" alt="Description" width="400" height="300">

Overall Review by products 7 zoom in to top 20 products:

<img src="https://hansuai-hong.github.io/assets/4.png" alt="Description" width="400" height="300">  <img src="https://hansuai-hong.github.io/assets/5.png" alt="Description" width="400" height="300">


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




### 6) Objective II - Sentiment Prediction (Random Forest Model)

To develop an effective sentiment prediction model, I implemented a Random Forest classifier using TF-IDF for feature extraction. The key steps involved in this process are as follows:

I applied the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer To transform text reviews into numerical features, selecting the top 5,000 features:

       vectorizer_tfidf = TfidfVectorizer(max_features=5000)
       X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
       X_test_tfidf = vectorizer_tfidf.transform(X_test)

To generate the model, i splited the data set to 80 train / 20 test and trained a Random Forest classifier with 100 decision trees and a fixed random state for reproducibility:

      rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
      rf_model.fit(X_train_tfidf, y_train)

The model was evaluated using accuracy, classification report, and a confusion matrix with visualiazation tools. also, the top 20most improtact features based on model's feature importance socred was displayed.

      y_pred = rf_model.predict(X_test_tfidf)
      accuracy = accuracy_score(y_test, y_pred)
      print(f"Accuracy: {accuracy:.4f}")
      print(classification_report(y_test, y_pred))

This is the initial models.

However, Upon evaluating the model, I observed a high number of false positives—cases where the model incorrectly predicted positive sentiment when the actual sentiment was negative. This issue suggested an imbalance in the dataset, as there were significantly more positive reviews than negative ones.

To improve model performance, I implemented the following adjustments:
Balancing the dataset: Since positive reviews outnumbered negative reviews, I duplicated negative remarks based on statistical guidelines until both classes had an equal number of samples.
Incresae the tree: Since number of tree is impacting the accuracy, I incerase to 200 for a better accuracy.
Retraining the model: After balancing the dataset and increase the tree, I retrained the Random Forest classifier, which improved its ability to differentiate between positive and negative sentiment.

These refinements enhanced the model’s predictive accuracy and reduced misclassification of negative reviews as positive.
the final results shows 95% accuracy which it is consier a good models now.



### 7) Objective II - Sentiment Prediction (RECURRENT NEURAL NETWORKS)

To enhance sentiment prediction, I implemented a 2nd model - Recurrent Neural Network (RNN), which is well-suited for analyzing sequential text data. 
Below is the steps to create the models:
1) Tokenization & Padding: Converting text reviews into numerical sequences and ensuring uniform input length.
2) # Split into training & validation sets
Model Architecture: Implementing an RNN-based model using LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit) layers to capture contextual meaning.
Training & Optimization: Using categorical cross-entropy loss and an Adam optimizer to train the model.
2. Model Evaluation
The RNN model was evaluated using:

Accuracy Score: To measure overall prediction performance.
Confusion Matrix: To analyze the distribution of correct and incorrect predictions.
Loss & Accuracy Plots: To monitor training performance and detect overfitting.
3. Challenges & Improvements
Overfitting Prevention: Applied techniques such as dropout regularization and early stopping to improve generalization.
Computational Cost: Optimized model parameters to improve training efficiency.
Comparison with Random Forest: Compared results with the previously trained Random Forest model to determine the best approach for sentiment classification.
The RNN model provided deeper contextual insights compared to traditional machine learning models, making it a valuable addition to sentiment analysis.





## Recommendation and Analysis
Explain the analysis and recommendations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
