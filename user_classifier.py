"""CSC110 Fall 2021 Final Project: user_classifier.py

This file is for the user to use the classifier as this would make the user interface a lot
simpler.
===============================

This file is provided solely for the personal and private use of staff
marking CSC110 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2021 Bryson Lo, Helena Glowacki, Emma Ho, and Chin Chin Jim."""

import pandas as pd
import numpy as np
from joblib import load
from real_or_fake_classifier import get_classifiers, prepare_x


class UserClassifier:
    """Class that aims to make trained classifier as easy to use as possible

    Attributes:
      - emotion_classifiers: dictionary of emotion attribute optimized classifiers
      - real_or_fake_classifier: real or fake optimized classifier

    Sample Usage:
    >>> classifier = UserClassifier(emotion_classifiers_folder_loaction =\
    '/emotion_classifiers', real_or_fake_classifier_location = 'rf_classifier.joblib')
    """

    emotion_classifiers: dict[str:GridSearchCV]
    real_or_fake_classifier: GridSearchCV

    def __init__(self, emotion_classifiers_folder_loaction: str, real_or_fake_classifier_location:
                 str) -> None:
        """Initialize a user classifier, loading all trained classifiers
        """
        self.emotion_classifiers = get_classifiers(emotion_classifiers_folder_loaction)
        self.rf_classifier = load(real_or_fake_classifier_location)

    def single_predict(self, tweet_content: str) -> dict[str:float]:
        """Return a dictionary that gives a prediction of wether or not the tweet text is fake or
         real and the probability of each outcome.

        >>> classifier = \
        UserClassifier(emotion_classifiers_folder_loaction= '/emotion_classifiers', \
        real_or_fake_classifier_location= 'rf_classifier.joblib')
        >>> sample_tweet_text = "micheal jackson thinks covid19 is not real and is a hoax!"
        >>> classifier.single_predict(sample_tweet_text)
        {'prediction': 'Fake', 'probability of real': 0.42365006842313113, 'probability of fake':
        0.5763499315768689}
        """

        # create blank dataframe
        df = pd.DataFrame()

        # format the tweet to have quotation marks (as all the fetched data from twitter were
        # wrapped in quotations)
        df["tweet_content"] = ["'" + tweet_content + "'"]

        # predict the emotion_attributes for this tweet and format it for prediction
        processed_input = [np.array(prepare_x(self.emotion_classifiers, df)).flatten()]

        # make predictions and store prediction result and proabilities in dictionary

        prediction_key = {1.0: 'Real', 0.0: 'Fake'}
        prediction_result = self.rf_classifier.predict(processed_input)[0]
        prediction_probabilities = self.rf_classifier.predict_proba(processed_input)[0]

        prediction_data = {
            'prediction': prediction_key[prediction_result],
            'probability of real': prediction_probabilities[1],
            'probability of fake': prediction_probabilities[0]
        }
        return prediction_data

    def batch_predict(self, tweets: list[str]) -> np.array:
        """Return list predictions of whether or not the tweet text is fake or real

        >>> classifier = \
        UserClassifier(emotion_classifiers_folder_loaction= '/emotion_classifiers', \
        real_or_fake_classifier_location= 'rf_classifier.joblib')
        >>> sample_tweet_list = ["micheal jackson thinks covid19 is not real and is a hoax!",\
        'The new coronavirus may spread more easily in crowded homes']
        >>> classifier.batch_predict(sample_tweet_list)
        ['Fake', 'Real']
        """
        # create blank dataframe
        df = pd.DataFrame()

        # format the tweets to have quotation marks (as all the fetched data from twitter were
        # wrapped in quotations)
        df["tweet_content"] = ["'" + tweet1 + "'" for tweet1 in tweets]

        # predict the emotion_attributes for this tweet and format it for training
        processed_input = np.array(prepare_x(self.emotion_classifiers, df))

        # reshape the input such that it is of the shape required for making predictions
        reshaped_input = []
        for tweet in range(len(tweets)):
            reshaped = []
            for feature in range(7):
                reshaped.append(processed_input[feature][tweet])
                reshaped_input.append(reshaped)

        prediction_key = {1.0: 'Real', 0.0: 'Fake'}

        prediction_result = self.rf_classifier.predict(np.array(reshaped_input))

        # return list of strings either of category 'real' or 'fake'
        return [prediction_key[res] for res in prediction_result]

    def make_prediction(self) -> None:
        """Allows the user to input a tweet or list or tweets and make predictions of wether
        or not they are real or fake
        """

        # Ask user to pick what prediction type they would like
        choice = input("Would you like to test a single tweet or a list of tweets? (single/list)")

        # Predict based on a single tweet
        if choice == "single":
            sample_tweet_text = input("Please enter your tweet")
            single_prediction = self.single_predict(sample_tweet_text)
            print('Single prediction results: ', single_prediction)

        # Predict based on a list of tweets
        elif choice == "list":
            sample_tweet_list = []
            choice_list = input("Please enter tweet 1 (If there are no more tweets enter 'end'):")
            while choice_list != 'end':
                sample_tweet_list.append(choice_list)
                choice_list = input("Please enter your next tweet (if there are no more tweets"
                                    " enter 'end'):")
            if sample_tweet_list == []:
                print("There were no tweets to test. Please try again")
            else:
                batch_prediction = self.batch_predict(sample_tweet_list)
                print('Batch prediction: ', batch_prediction)
        else:
            print("User did not enter correct input. Expected input 'single' or 'list'. Please "
                  "try again.")


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['make_prediction'],
        'max-line-length': 100,
        'extra-imports': ['python_ta.contracts', 'pandas', 'numpy', 'joblib',
                          'real_or_fake_classifier'],
        'disable': ['R1705', 'C0200'],
    })

    import doctest
    doctest.testmod()
