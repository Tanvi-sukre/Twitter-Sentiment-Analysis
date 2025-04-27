# Twitter-Sentiment-Analysis
A Machine Learning and NLP project that classifies tweets as positive or negative using the Sentiment140 dataset. Preprocessing, TF-IDF vectorization, and various ML models were applied, with Logistic Regression achieving 75.42% accuracy. SVM and Gradient Boosting were explored but skipped due to long computation times.

A project to classify sentiments (positive or negative) of tweets using Natural Language Processing (NLP) and Machine Learning techniques.

📚 Project Overview
In this project, we:

Collected a dataset of 1.6 million tweets (Sentiment140)

Preprocessed the text (cleaning, tokenization, stemming, removing stopwords)

Converted tweets into numerical features using TF-IDF Vectorization

Trained and evaluated multiple Machine Learning models to classify tweet sentiments

Visualized word clouds, tweet length distribution, and model performances

🛠 Tech Stack
Python

NLTK (Natural Language Processing Toolkit)

Scikit-learn (ML models and TF-IDF)

Pandas (Data manipulation)

Matplotlib / Seaborn (Visualization)

📂 Dataset
Source: Sentiment140 Dataset

Size: 1,600,000 tweets

Fields: Target (Sentiment), User ID, Date, Query, User, Tweet Text

🧪 Models Trained
Logistic Regression ✅ (Best Performing - 75.4% Accuracy)

Random Forest

Naïve Bayes

Decision Tree

(Attempted SVM and Gradient Boosting, but skipped due to high computation time on full dataset)

🛠 Preprocessing Steps
Removal of non-alphabetic characters

Lowercasing all text

Tokenization

Stopword removal

Stemming

TF-IDF feature extraction

📈 Results

Model	                  Accuracy	  Precision	   Recall
Logistic Regression	     75.42%	      0.76	      0.74
Random Forest	           74.34%	      0.75	      0.73
Naïve Bayes	             74.06%	      0.74	      0.72
Decision Tree	           69.68%	      0.70	      0.68
🏆 Final Model: Logistic Regression was selected for its balance of accuracy, speed, and simplicity.

📊 Visualizations
Word Clouds for positive and negative tweets

Tweet Length Distribution

Confusion Matrix for each model

Accuracy Comparison Bar Chart

⚠️ Notes
SVM and Gradient Boosting models were tested on a subset but took over 4 hours on the full dataset, so they were excluded from the final full-dataset evaluation for efficiency.

Focused on binary classification (Positive vs Negative). Extension to a 3-class (Positive, Neutral, Negative) is possible.

🚀 Future Work
Implement deep learning models (e.g., LSTM, BERT)

Extend to multi-class classification (Neutral sentiment)

Deploy a real-time sentiment analysis web app using Streamlit or Flask

Deploy on cloud platforms (AWS, GCP, Azure)

