"""CSC110 Fall 2021 Final Project: main.py

This file is our main file and contains the code necessary to run our program
This file loads the necessary files from the datasets performs computations on them,
and produces a non-interactive output.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of staff
marking CSC110 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2021 Bryson Lo, Helena Glowacki, Emma Ho, and Chin Chin Jim.
"""
from tweet_scraping import scrape_and_process_data
from emotion_classifiers import create_emotion_attr_classifier_files
from real_or_fake_classifier import create_real_or_fake_classifier_file
from user_classifier import UserClassifier

# 1. Scrape tweets using Tweepy api, combine it with original data and create two formatted dataset
# files
# (one for training/testing the emotion attributes classifier and the other for training/testing
# the real or fake classifier)

# scrape_and_process_data()

# 2. Create the emotion attribute classifiers

# create_emotion_attr_classifier_files(emotion_data_path = 'emotion.csv')

# 3. Use emotion attribute classifiers on kaggle data and create a real/fake tweet classifier

# create_real_or_fake_classifier_file(kaggle_data_path = 'kaggle.csv')

# 4. Use classifiers to create new UserClassifier instance

classifier = UserClassifier(
  emotion_classifiers_folder_loaction='/emotion_classifiers',
  real_or_fake_classifier_location='rf_classifier.joblib')

# 5. Allow the user to make real/fake predictions on a single tweet or a list of tweets
classifier.make_prediction()
