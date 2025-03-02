---
layout: post
author: HanSuai, Hong
title: "Product Recommendation System for Sephora Online Store"
categories: ITD214
---
# Project Background 

This work is a part of a group project module from Nanyang Polytechnic Singapore's Specialist Diploma in Business & Big Data Analytics.
- Project module : ITD214 - Applied Data Science Project
- Group 5 : Applying Data Science to Improve/Optimize Skincare Business Model Through Customer Reviews

The beauty industry is constantly evolving, with shifting consumer preferences, emerging market trends, and increasing competition. To maintain a strong market position, Sephora Products must leverage on AI strategies to enhance customer experience and drive its business growth.

This project focuses on utilizing customer insights, product performance data, and market trends for SEPHORA to gain a deeper understanding of customer behavior and industry dynamics. By analyzing purchasing patterns, customer feedback, and engagement metrics, Sephora can make use of those data to improve product offerings, optimize marketing strategies, and refine pricing structures. A data-driven approach ensures the brand remains responsive to consumer demands while maximizing operational efficiency and profitability.
<p align="center">
<img src="https://hansuai-hong.github.io/assets/0.jpg" alt="Description" width="600" height="450">
</p>

# Project Objective
This project is driven by 4 key objectives, each led by a dedicated team member:

1) Sentiment Analysis –  
   Analyze customer reviews, feedback, and social media sentiment to understand customer perceptions of Sephora’s products. This insight helps identify areas for improvement and enhance customer satisfaction.
   
2) Customer Segmentation and Preference Analysis –  
   Categorize customers based on demographics, purchasing behavior, and preferences to create targeted marketing strategies and personalized experiences.
   
3) Develop a Product Recommendation System –    
   Build a recommendation engine using machine learning to suggest relevant products based on customer preferences and browsing history, enhancing user engagement and sales.
   
4) Price Optimization –   
   Implement data-driven pricing models that consider demand patterns, competitor pricing, and customer willingness to pay, ensuring competitive yet profitable pricing strategies.

By integrating these 4 analytical approaches, we believe that Sephora Products can be improved to another stage in term of individual personalization, customer loyalty, and increase in business profitability.


## Personal Objective


<p align="center">
<img src="https://hansuai-hong.github.io/assets/1.png" alt="Description" width="600" height="450">
</p>

## Hong Han Suai : SENTIMENT ANALYSIS
Aim to extract meaningful insights from customer reviews and develop a predictive model to classify sentiments effectively. This is achieved through two key objectives:

1) Understanding Customer Sentiment -  
Analyze review text to identify the most common words used across all reviews.
Determine the most frequently mentioned words in positive and negative reviews separately.
Use word clouds to visually represent the pros and cons of Sephora products based on customer feedback.

3) Building a Sentiment Prediction Model -  
Develop a machine learning model to predict sentiment based on customer reviews.  
two different approaches:  
      - Random Forest: A traditional machine learning model that leverages decision trees to classify reviews as positive or negative.  
      - Recurrent Neural Network (RNN): A deep learning model designed to capture contextual meaning from text data for more accurate sentiment prediction.  

These objectives aim to provide Sephora with valuable insights into customer perceptions while enabling automated sentiment classification for future reviews.



# Work Accomplished
### 1) Data Collection (Group effort)
As a team, we conducted thorough research to identify a suitable dataset for our analysis. After evaluating multiple sources, a dataset from Kaggle was chosen that provides comprehensive insights relevant to our project objectives. 

This dataset includes two key components: 

Info Data - contains general SEPHORA product details  
Review Data - consists of five sheets of customer reviews and associated information.

After obtaining the dataset, an initial Exploratory Data Analysis (EDA) was generated to gain a better understanding of its structure and quality. This process involved:
- Reviewing data distribution.
- Identifying inconsistencies, missing values, and duplicate entries.
- Detecting outliers.
- Recognizing key patterns to guide deeper analysis.

This preliminary analysis helped ensure that the dataset was well-prepared for subsequent sentiment analysis and predictive modeling tasks.
   

## 2) Exploratory Data Analysis (EDA)
To perform sentiment analysis, a few key parameters were identified such as Product ID, Rating, and Text Review to incooperate to sentiment analysis. Before proceeding, a simple Exploratory Data Analysis (EDA) was conducted to understand the dataset's structure and distribution on the parameters which identified

Overall Rating Distribution:
<p align="center">
<img src="https://hansuai-hong.github.io/assets/2.png" alt="Description" width="400" height="300">
</p>
Positive reivew (rating-4 and rating-5)
negative review (rating-1, rating-2, and rating-3)

      df_reviews['sentiment'] = df_reviews['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')
<p align="center">
<img src="https://hansuai-hong.github.io/assets/3.png" alt="Description" width="400" height="300">
</p>
Overall Review Counts by products and zoom in to top 20 products:
<p align="center">
<img src="https://hansuai-hong.github.io/assets/4.png" alt="Description" width="400" height="300">  <img src="https://hansuai-hong.github.io/assets/5.png" alt="Description" width="400" height="300">
</p>

## 3) Data Cleaning
To ensure data accuracy and consistency, the following cleaning tasks were done:
- Removing missing values (which less than 1% of total data set).
- Removing duplicate entries to prevent redundancy in the dataset.
- Convert the submission time to datetime format.
- Eliminate irrelevant columns that do not contribute to sentiment analysis.
 
      df_reviews1 = df_reviews.dropna(subset=['review_text'])
      df_reviews1['submission_time'] = pd.to_datetime(df_reviews1['submission_time'])
      df_reviews2 = df_reviews1.drop_duplicates()
      df_reviews3 = df_reviews2[['review_text' ,'rating', 'product_id']
       

## 4) Data preprocessing
after cleaning, next step will be data preprocessing which include:

- Standardizing text formats by converting all text to lowercase for consistency.
- Punctuation and Special Character Removal: Cleaning text by eliminating unnecessary symbols.
- Tokenization: Splitting customer reviews into individual words.
- Stopword Removal: Removing common words that do not provide meaningful insights.
- Lemmatization: Converting words to their root form for consistency.

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

## 5) Filtering  
Since performing sentiment analysis on 1 million reviews across 8000 products is too generic, ipywidgets was introduced to allow filter results by Product ID. Once a specific product is selected, user then analyze its reviews in detail.

      # Create an interactive widget
      interactive_widget = widgets.interactive(filter_reviews, product_id=product_input)

      # Display the input widget
      display(interactive_widget)
<p align="center">      
<img src="https://hansuai-hong.github.io/assets/7.png" alt="Description" width="300" height="75">
</p>


## 6) Objective A - Understand Customer Sentiment with Key Insights
All reviews were analyzed on the particular sekected products to identify the most commonly used words. This helps determine recurring themes and patterns across the dataset.

      # calculate the number of occurence of each word in the entire list of words
      all_words_frequency = FreqDist(all_words)
      print (all_words_frequency)

      # print 10 most frequently occurring words
      print ("\nTop 20 most frequently occurring words")
      print (all_words_frequency.most_common(20))
This is the most common 20words used in the particular selected product using Hbar:
<p align="center">  
<img src="https://hansuai-hong.github.io/assets/6.png" alt="Description" width="450" height="320">
  </p>

This is the most common 20words used in the particular selected product using WordCloud:
<p align="center">  
<img src="https://hansuai-hong.github.io/assets/9.png" alt="Description" width="600" height="450">
</p>

To gain deeper insights, reviews were categorizd into positive and negative sentiments. By isolating the most frequently mentioned words in positive reviews, the top 10 frequent words of particular products from positive were highlighted. Similarly, negative reviews also been pinpointed from 10 most frequent words of particular products. Nevertheless, those frequent words appear in both postive and negative (common words) are filtered out to improve the effectiveness.

Word clouds were used to represent the pros and cons of Sephora products based on customer feedback to effectively visualize the findings. Word clouds provide an intuitive way to showcase prominent words, making it easier to recognize key attributes associated with customer feedbacks. This approach helps to extract meaningful insights from large volumes of review data, ultimately contributing to a better understanding of customer preferences and areas for product enhancement.
<p align="center">  
<img src="https://hansuai-hong.github.io/assets/10.png" alt="Description" width="1000" height="500">
</p>

## 7) Objective B1 - Sentiment Prediction (Random Forest Model)

To develop an effective sentiment prediction model, a Random Forest classifier using TF-IDF for feature extraction was inplemented. The key steps involved in this process are as follows:

1) TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer To transform text reviews into numerical features, selecting the top 5,000 features:

       vectorizer_tfidf = TfidfVectorizer(max_features=5000)
       X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
       X_test_tfidf = vectorizer_tfidf.transform(X_test)

2) data set splitting (80% train / 20% test):
  
       X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
   
3) Random Forest classifier generation with 100 decision trees and a fixed random state for reproducibility:

       rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
       rf_model.fit(X_train_tfidf, y_train)

4) The model was build and evaluated using accuracy, classification report, and a confusion matrix with visualiazation tools. also, the top 20 most important features based on model's feature importance scored was displayed.

      y_pred = rf_model.predict(X_test_tfidf)
      accuracy = accuracy_score(y_test, y_pred)
      print(f"Accuracy: {accuracy:.4f}")
      print(classification_report(y_test, y_pred))
<p align="center"> 
<img src="https://hansuai-hong.github.io/assets/11d.png" alt="Description" width="800" height="500">
</p>

This is the initial models.

However, Upon evaluating the model, it was observed that a high number of false positives—cases where the model incorrectly predicted positive sentiment when the actual sentiment was negative. This issue suggested an imbalance in the dataset, as there were significantly more positive reviews than negative ones.

To improve the model performance, the following adjustments were introduced:  
- Balancing the dataset:  
Since positive reviews outnumbered negative reviews, negative remarks have been duplicated based on statistical guidelines until both classes had an equal number of samples.  
- Incresae the number of tree:   
Since number of tree is impacting the accuracy, from initial 100 trees, it is extended to 200 trees for a better accuracy.  

      avg_count = int((len(df_positive) + len(df_negative)) / 2)

      # 5️Resample Both Classes to Match the Average Count (Only if needed)
      df_positive_balanced = resample(df_positive, replace=True, n_samples=avg_count, random_state=42)
      df_negative_balanced = resample(df_negative, replace=True, n_samples=avg_count, random_state=42)

      # 6️Combine the Balanced Dataset
      filtered_balanced = pd.concat([df_positive_balanced, df_negative_balanced])

      # 7️Shuffle the Dataset
      filtered_balanced = filtered_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
      rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
      rf_model.fit(X_train, y_train)

These refinements enhanced the model’s predictive accuracy and reduced misclassification of negative reviews as positive.
the final results shows 95% accuracy which it can be used for predictive on future reviews.

<p align="center"> 
<img src="https://hansuai-hong.github.io/assets/12d.png" alt="Description" width="800" height="500">
</p>


## 8) Objective B2 - Sentiment Prediction (Recurrent Neural Network)

To enhance sentiment prediction, 2nd model - Recurrent Neural Network (RNN) was inplemented, which is well-suited for this scenario. 
Below is the steps to create the models:  

1) Tokenization & Padding: Converting text reviews into numerical sequences and ensuring uniform input length.  
2) Split into training & validation sets: split the data to 80% train data and 20% test data.  
3) Model Architecture:    
      a. Bidirectional LSTM Layer: Captures long-range dependencies in both forward and backward directions using 128 LSTM units with L2 regularization.  
      b. GRU Layer: Processes the sequential data further with 64 GRU units and L2 regularization.  
      c. Dropout Layer: A dropout rate of 0.5 is applied to prevent overfitting.  
      d. Dense Layers: The output from the GRU layer is passed through a fully connected dense layer with 64 neurons and ReLU activation.  
      e. Output Layer: A final dense layer with a sigmoid activation function predicts the probability of a review being positive.

       # Convert text to sequences
       X_sequences = tokenizer.texts_to_sequences(filtered_final['cleaned_review_text'])

       # Pad sequences
       max_length = 100  # Fixed max length
       X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

       # Convert labels to numpy array
       y = np.array(filtered_final['sentiment'])

       # Split into training & validation sets
       X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, random_state=42)

       # Model Hyperparameters
       embedding_dim = 100
       lstm_units = 128
       gru_units = 64
       dropout_rate = 0.5
       l2_reg = 0.01
       batch_size = 64
       epochs = 20
       vocab_size = 10000  # Same as tokenizer's num_words

       # Model Architecture
       model = Sequential([
       Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, trainable=True),
       Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg))),
       GRU(gru_units, return_sequences=False, kernel_regularizer=l2(l2_reg)),
       Dropout(dropout_rate),
       Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
       Dense(1, activation='sigmoid')
       ])
   
4) Training & Testing: Using categorical cross-entropy loss and an Adam optimizer to train the model. 

        # Compile Model
        model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


        # Train Model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),batch_size=batch_size, epochs=epochs, callbacks=[early_stop, reduce_lr, checkpoint])

5) Callbacks function: Using early_stop and recued_Ir to reduce the number of epochs when the accuracy reach it saturated point.

       # Callbacks
       early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
        checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    

6) The RNN model was evaluated using accuracy score, confusion matrix and loss & accuraccy plots to see how accurate the model is:  
     a. Accuracy Score: To measure overall prediction performance.  
     b. Confusion Matrix: To analyze the distribution of correct and incorrect predictions.  
     c. Loss & Accuracy Plots: To monitor training performance and detect overfitting.  

<p align="center"> 
<img src="https://hansuai-hong.github.io/assets/13d.png" alt="Description" width="800" height="500">
</p>

The initial accuracy is not bad (73%). However, all prediction are positive reviews. Something is not correct here. To correct the error, modofication was done such as:
  - balanced the positive and negative reviews (same as what I did for the random forest model)  
  - Finetune parameters such as embedding_dim, lstm_units, gru_units, dropout_rate and l2_reg, vocab_size  
  - add in extra one Dense layer  

<p align="center"> 
<img src="https://hansuai-hong.github.io/assets/14d.png" alt="Description" width="800" height="500">
</p>


# Summary and Future Improvements
Totaly 2 Objective for personal Goal for personal (Sentiment Analysis) :

#### 1) Objective 1 - Understand Customer Sentiment with Key Insights
Summary:  
A model is inplemented succesfully using wordclouds to recognize key attributes associated with customer satisfaction and dissatisfaction for a particular   product with user input inerface.

Recommendation & improvement:  
further improment can be done by not only using product ID, we can filter by brand, catagory, human attributes and etc to group the data. Also, we can filter or unfilter certain key words by human input instead of auto filter to make sure no important message has been left out.

        
#### 2) Objective 2 - Sentiment Prediction  
Summary:  
2 models were build. 1st model was based on traditional Random forest approach and 2nd model was baesd on RNN deep learning approach. both models can achieved >90% accuracy after fine tuning and improvement.

Recommendation & improvement:  
Further improvemetn can be done as using the predictive result to replace current data set which rating and is_recommended is empty. Also, the coding can be further improved by adding in the new data set to existing data set to retrain to achieve self learning and self improvemnt to imporve the accuracy.
    

# AI Ethics
AI Ethical which will take into considerations for Sentiment Analysis in Sephora’s Business Development include:

1) Fairness and Bias  
A fundamental ethical challenge in sentiment analysis is the risk of bias in AI models. As the dataset is unbalanced, the model may unfairly favor to certain products or customer sentiments, cause misleading insights. For example, if the majority reviews are positive, the model might struggle to identify negative feedback accurately. To address this, techniques such as dataset balancing should be used to ensure fair representation of all customer opinions. Others than data bias, we should also not unintentionally discriminate based on factors such as product categories, demographics, or review sources.

2) Transparency and Explainability  
Another challenge for AI models is black boxes phenomenon. The model must enssure that the outcome and prediction are explainable and interpretable. In the case of Random Forest models, feature importance scores can highlight which words contribute most to sentiment classification. For deep learning models like RNNs, methods such as LIME (Local Interpretable Model-Agnostic Explanations) or SHAP (Shapley Additive Explanations) can help explain predictions. When presenting sentiment analysis findings, businesses must clearly communicate model accuracy, limitations, and the reliability of predictions to avoid any misleading and misunderstanding.

3) Privacy and Data Protection  
Customer reviews often contain sensitive information, making privacy protection a critical ethical responsibility. When performing sentiment analysis, we should anonymize personal data to prevent the identification of individual customers been leaked out. We must also ensure compliance with data protection laws such as GDPR (General Data Protection Regulation) PDPA (Personal Data Protection Act). 

4) Ethical Business Use  
The insights derived from sentiment analysis must used ethically to enhance customer experience rather than manipulate perceptions. We must avoid using AI findings to mislead consumers by exaggerating positive sentiment. Instead, businesses should leverage sentiment insights to improve product quality, refine marketing strategies, and provide better customer service. Ethical considerations should also guide marketing campaigns, ensure that the insights are used to create honest and transparent advertising not the other way round.

AI models require continuous monitoring to ensure they remain accurate and ethical over time. Sentiment trends can evolve, and models trained on past data may become outdated or biased. Thus, regular performance tracking, bias audits, and model retraining will help and support in reliability. Additionally, human oversight is essential to review AI-generated insights and prevent potential misinterpretations. Businesses should establish a clear process for addressing errors in sentiment classification and ensure that AI-driven decisions are always comply to those 4 ethics mentioned above.


## Source Codes and Datasets

Data set:  Data source file link (Kaggle) : https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews  
Source Codes a@ Github REPO : https://github.com/Hansuai-Hong/ITD214
