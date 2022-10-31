"""CSC110 Fall 2021 Final Project: emotion_classifiers.py

This file use identified emotion metrics of tweets from the COVID-19 Twitter Dataset from
OpenICPSR to train emotion metric classifiers. Emotion metric classifiers determine the emotion
metrics of a tweet based on its content

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of staff
marking CSC110 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2021 Bryson Lo, Helena Glowacki, Emma Ho, and Chin Chin Jim."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from joblib import dump


def create_emotion_attr_classifier_files(emotion_data_path: str) -> None:
    """Create all emotion attribute classifiers and save them to destination folder
    """

    # 1. get current file directory
    current_dir = os.getcwd() + '/'
    path = os.path.join(current_dir, 'emotion_classifiers')

    # 2. attempt to make new folder named 'emotion_classifiers'. An error will be raised if the
    # folder already exists
    try:
        os.mkdir(path)
    except OSError as error:
        print('directory already exists')

        # 3. preprocess and format data
    training_set, test_set = data_preprocessing(emotion_data_path)

    # 4. the following are the variables we wish to make classifiers for
    target_variables = ["valence intensity", "sadness intensity", "happiness intensity",
                        "anger_intensity", "fear_intensity", "emotion", "sentiment"]

    # 5. save optimized classifiers to folder
    for var in target_variables:
        classifier = EmotionAttrClassifier(var, training_set, test_set)
        classifier.fit_classifier()
        classifier.optimize_classifier()
        classifier.save_optimized_clf_to_file(save_location='./emotion_classifiers')


def data_preprocessing(emotion_data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess data and split it into training and testing sets
    """
    # 1. load data from file location
    data = pd.read_csv(emotion_data_path)

    # 2. remove all observations of dataset where tweet content failed to be fetched
    data = data[data['tweet_content'] != '\'None\'']

    # thes following intensities are given variables in the COVID-19 Twitter Dataset
    intensities = ['valence', 'fear', 'anger', 'happiness', 'sadness']

    # 3. categorize the continous numerical intensity variables to 5 different variables:
    # 'very low', 'low', 'medium', 'high', 'very high'.
    # Since intensity range between 0 to 1. Values in interval [0, 0.2) are 'very low',
    # [0.2, 0.4) are 'low', [0.4, 0.6) are 'medium', [0.6, 0.8) are 'high' and [0.8,1.0] are '
    # very high'

    for intensity in intensities:
        emotion_intensity = intensity + "_intensity"
        data[emotion_intensity] = pd.cut(data[emotion_intensity],
                                         bins=5,
                                         labels=['very low', 'low', 'medium', 'high', 'very high'])

    # 4. split up the data into 80% training and 20% testing
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=22)

    return train_set, test_set


class EmotionAttrClassifier:
    """An emotion attribute classifier

    Attributes:
      - clf: the classifier (which may not have been fitted (trained with) training data)
      - grid_search_clf: the optimized classifier using GridSearchCV
      - target_variable: the emotion metric this classifier is being trained to predict
      - training_set: set of training data that will be used to train classifier
      - testing_set: set of testing data that will be used to test classifier

    Representation Invariants:
      - self.target_variable in ["valence intensity", "sadness intensity",
      "happiness intensity", "anger_intensity", "fear_intensity", "emotion", "sentiment"]

    #
    """

    clf: Pipeline
    grid_search_clf: GridSearchCV
    target_variable: str
    traing_set: pd.DataFrame
    testing_set: pd.DataFrame

    def __init__(self, target_variable: str, training_set: pd.DataFrame, testing_set: pd.DataFrame):
        """Initialize an emotion attribute classifier

        The classifier begins with an unfitted model (self.clf) and no optimized classifier
        """
        self.clf = Pipeline([
            ('CV', CountVectorizer()),
            ('TT', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge',
                                  penalty='l2',
                                  alpha=1e-3,
                                  random_state=22,
                                  max_iter=5,
                                  tol=None))])
        self.grid_search_clf = None
        self.target_variable = target_variable
        self.training_set = training_set
        self.testing_set = testing_set

    def fit_classifier(self) -> Pipeline:
        """Fit the model to training data
        """
        y_train = np.array(self.training_set[self.target_variable])
        x_train = self.training_set["tweet_content"]
        self.clf.fit(x_train, y_train)
        return self.clf

    def optimize_classifier(self) -> GridSearchCV:
        """Optimizes the model using GridSearchCV
        """
        # the following are the parameters that will be tested using GridSearchCV
        # to find the best combination of parameters that gives the classifier the best performance
        parameters = {
            'CV__ngram_range': [(1, 1), (1, 2)],
            'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5),
            'clf__max_iter': (100, 150, 200, 250)
        }
        y_train = np.array(self.training_set[self.target_variable])
        x_train = self.training_set["tweet_content"]

        # create an instance of GridSearchCV to find the optimal parameters for the logistic
        # regression model
        self.grid_search_clf = GridSearchCV(self.clf, parameters, cv=5, n_jobs=-1)

        # fit the optimized model to our training data
        self.grid_search_clf.fit(x_train, y_train)
        return self.grid_search_clf

    def optimized_accuracy(self) -> np.array:
        """Returns the accuracy of our classifer
        """
        Y_test = np.array(self.testing_set[self.target_variable])
        X_test = self.testing_set["tweet_content"]
        # make a prediction on our test data
        predicted = self.grid_search_clf.predict(X_test)
        # return the accuracy of our the predictions
        return np.mean(predicted == Y_test)

    def save_optimized_clf_to_file(self, save_location: str) -> None:
        """Saves optimized classifer to a file at desired location
        """
        file_path = f'{save_location}/{self.target_variable}_clf.joblib'
        dump(self.grid_search_clf, file_path)


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['create_emotion_attr_classifier_files'],
        'max-line-length': 100,
        'extra-imports': ['python_ta.contracts', 'pandas', 'numpy', 'joblib', 'sklearn.pipeline',
                          'sklearn.model_selection', 'sklearn.linear_model',
                          'sklearn.feature_extraction.text', 'os'],
        'disable': ['R1705', 'C0200'],
    })
