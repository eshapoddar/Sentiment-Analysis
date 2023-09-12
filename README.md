# Amazon Food Reviews Sentiment Analysis
This repository contains code and documentation for a sentiment analysis project using the Amazon Fine Food Reviews dataset. The project aims to predict whether Amazon food product reviews are positive, neutral, or negative based on the review text. It includes several machine learning models, including logistic regression, support vector machines, and a BERT-based model.

## Dataset
**Chosen Dataset**

We used the Amazon Fine Food Reviews dataset, available on Kaggle here. This dataset contains over 500,000 reviews of food products sold on Amazon from 2002 to 2012. It includes various information such as product ID, user ID, score, summary, text review, timestamp, and more.

The dataset was chosen for its size, variety of products, and reviews, making it suitable for natural language processing (NLP) tasks like sentiment analysis and topic modeling.

**Labels and Text Input**

Labels: The product ratings (Score) from 1 to 5 are the original labels. We created a new set of target labels ranging from 0 to 2 for classifiers and the BERT model.

Mapping:

0: Negative review (Score 1 or 2)

1: Neutral review (Score 3)

2: Positive review (Score 4 or 5)

Text input: The input text used in the models is a combination of the text and summary columns, preprocessed to obtain TF-IDF word vectors.

**Training, Validation, and Test Split**

The dataset was split into training (60%), validation (20%), and test (20%) sets using scikit-learn's train_test_split function.

The label distribution is fairly balanced across splits, though there are more high-rating reviews (4 and 5) than low-rating ones (1, 2, and 3).

## Clustering
We performed clustering using k-means with k=5 clusters and identified top tokens and documents for each cluster. The clusters appeared to capture different topics, such as coffee and tea, hot chocolate, general food products, gluten-free mixes, and potato chips.

## Comparing Classifiers
We compared the performance of several classifiers on both the training and validation sets, using both one-hot and TF-IDF vectorization. The best-performing classifier on the validation set was Logistic Regression using Count Vectorization, achieving an F1 score of 0.578.

## Parameter Tuning
Parameter tuning was performed for Logistic Regression with TF-IDF vectorization. The best parameters found were C=1000, max_features=None, solver='sag', and sublinear_tf=False, resulting in an optimized F1 score of 0.536 on the validation set.

## Context Vectors using BERT
We utilized BERT-based context vectors to train a logistic regression classifier. Multiple models with different hyperparameters were tested, and Model2 with learning_rate=2e-5, batch_size=16, and num_train_epochs=5 outperformed the others on the validation set.

## Conclusions and Future Work
The best-performing approach achieved an accuracy of 0.89 on the test set. However, there is room for improvement, particularly in handling neutral reviews. Future work could focus on collecting more diverse and balanced data, fine-tuning hyperparameters, and addressing biases in the training data. Additionally, model interpretability and ethical considerations should be explored when deploying such systems.
