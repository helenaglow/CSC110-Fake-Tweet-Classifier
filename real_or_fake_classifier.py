"""CSC110 Fall 2021 Final Project: real_or_fake_classifier.py

We used our emotions classifier to classify the emotion metrics of each tweet from our COVID-19
Fake News Dataset found on Kaggle, and used it to train this real_or_fake_classifier to classify
whether tweets are real or fake based on emotion metrics.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of staff
marking CSC110 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2021 Bryson Lo, Helena Glowacki, Emma Ho, and Chin Chin Jim."""

import pandas as pd
import numpy as np
from joblib import dump, load

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# these variables are what we aim to generate for our observations of the kaggle dataset
VARIABLES = ['valence_intensity',
             'fear_intensity',
             'anger_intensity',
             'happiness_intensity',
             'sadness_intensity',
             'emotion',
             'sentiment']


def create_real_or_fake_classifier_file(kaggle_data_path: str) -> None:
    """Main function that saves the twitter real/fake classifier to a file"""

    # 1. loads the kaggle dataset containing tweet text that was scraped off twitter
    unprocessed_data = load_combined_kaggle_data(kaggle_data_path)

    # 2. retrieves the trained emotion metric classifier files from /emotion_classifiers folder
    classifiers = get_classifiers(classifiers_folder_location='./emotion_classifiers')

    # 3. apply emotion metric classifiers to data to create new variables and format for tarining
    train_set, test_set = data_preprocessing(
        prepare_x(classifiers, unprocessed_data),
        prepare_y(unprocessed_data)
    )

    # 4. train the classifier with formatted data and save classifier to a file
    train_and_save_classifier(train_set, test_set)


def load_combined_kaggle_data(file_location: str) -> pd.DataFrame:
    """Loads the kaggle dataset containing tweet text that was scraped off twitter"""

    # 1. load data from file location
    data = pd.read_csv(file_location)

    # 2. remove all observations of dataset where tweet content failed to be fetched
    data = data[data['tweet_content'] != '\'None\'']

    # 3. remove all duplicate tweets
    data = data.drop_duplicates(subset=["tweet_content"]).reset_index(drop=True)

    return data


def get_classifiers(classifiers_folder_location: str) -> dict[str:GridSearchCV]:
    """Returns a dictionary of trained emotion metric classifiers from its location"""
    clfs = {}

    # store each classifier into a dictionary with corresponding variable keys
    for var in VARIABLES:
        filename = f'./{classifiers_folder_location}/{var}_clf.joblib'
        clfs[var] = load(filename)

    return clfs


def prepare_x(classifiers: dict[str:GridSearchCV], data: pd.DataFrame) -> list[np.array]:
    """Use emotion metric classifiers to create new variables for dataset and format it
     for training"""
    attributes = []

    # using each emotion metric classifier, make predictions for each observation in the dataset
    for var in VARIABLES:
        attributes.append(np.c_[classifiers[var].predict(data["tweet_content"])])

    # to train the the new classifier, numerical values will be needed,
    # hence the categories will be encoded using an ordinal encoder
    ordinal_attributes = []

    # loop through the first five variables are all intensities and have the same categories
    for attribute in attributes[:-2]:
        # below are the 5 possible categories for intensities
        # (the categories must be specified as some categories do not contain any observations
        # in our sample of the dataset and thus the ordinal encoder will only encode to 4 categories
        # if categoires weren't specified)

        cats = [['very high', 'high', 'medium', 'low', 'very low']]
        # the ordinal encoder then encodes these categories from 0 - 5 with 'very high' being 0 and
        # 'very low' being 5
        ordinal_encoder = OrdinalEncoder(categories=cats)
        ordinal_attributes.append(ordinal_encoder.fit_transform(attribute.copy()).flatten())

    # our sample from the dataset contains observations from all categories of the emotion variable,
    # thus the categories do not need to be specified
    emotion_ordinal_encoder = OrdinalEncoder()
    ordinal_attributes.append(emotion_ordinal_encoder.fit_transform(attributes[-2]).flatten())

    # similarly to the intensity encoding, the categories must be specified as some categories do
    # not contain any observations in our sample of the dataset and thus the ordinal encoder will
    # only encode to 4 categories if categoires weren't specified
    sentiment_cats = [['very negative', 'negative', 'neutral or mixed', 'positive',
                       'very positive']]
    sentiment_ordinal_encoder = OrdinalEncoder(categories=sentiment_cats)
    ordinal_attributes.append(sentiment_ordinal_encoder.fit_transform(attributes[-1]).flatten())

    return ordinal_attributes


def prepare_y(data: pd.DataFrame) -> np.array:
    """Encode real_or_fake variable to 0s and 1s to format it for training"""

    real_fake_categories = data[["real_or_fake"]]
    ordinal_encoder = OrdinalEncoder()

    real_fake_encoded = ordinal_encoder.fit_transform(real_fake_categories)

    return real_fake_encoded.flatten()


def data_preprocessing(x: list[np.array], y: np.array) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create training and testing sets to train and test the classifier"""
    prepped_targets = y
    prepped_attributes = x

    # create a blank dataframe
    data_prepped = pd.DataFrame()

    # add each emotion metric variable for the observations to the dataframe
    for var_index in range(len(VARIABLES)):
        data_prepped[VARIABLES[var_index]] = prepped_attributes[var_index].tolist()

    # add encoded version of real_or_fake for the observations to the dataframe
    data_prepped["encoded_real_fake"] = prepped_targets.tolist()

    # split new formatted data into test and training data
    # 80% of the observations in the dataset are used for training and 20% for testingap
    # the data is shuffled to ensure that we have a reasonable amount of fake and real observations
    # for training
    train_set, test_set = train_test_split(data_prepped, test_size=0.2, random_state=22,
                                           shuffle=True)

    return train_set, test_set


def train_and_save_classifier(train_set: pd.DataFrame, test_set: pd.DataFrame) -> GridSearchCV:
    """Train the classifier with formatted data and save optimized classifier to a file
    """

    # split data into desired inputs X and desired outputs Y
    x_train = train_set.copy().drop(columns="encoded_real_fake")
    x_test = test_set.copy().drop(columns="encoded_real_fake")
    y_train = train_set.copy()["encoded_real_fake"]
    y_test = test_set.copy()["encoded_real_fake"]

    # create an instance of the LogisitcRegression model
    clf = LogisticRegression(random_state=22, solver='liblinear')

    # fit the model to our training data
    clf.fit(x_train, y_train)

    # make a prediction on our test data
    predict = clf.predict(x_test)
    # output the accuracy of our non-optimized classifier
    print(f'accuracy of {np.mean(predict == y_test)} on test data!')

    # the following are the parameters that will be tested using GridSearchCV
    # to find the best combination of parameters that gives the classifier the best performance
    parameters = {
        'C': (0.01, 0.05, 0.1, 0.5, 1.0),
        'tol': (1e-3, 1e-4, 1e-5),
        'max_iter': (100, 200, 300),
        'warm_start': (True, False)
    }

    # create an instance of GridSearchCV to find the optimal parameters for the logistic regression
    # model
    optimized_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)

    # fit the optimized model to our training data
    optimized_clf.fit(x_train, y_train)

    # make a prediction on our test data
    opt_predict = optimized_clf.predict(x_test)
    # output the accuracy of our optimized classifier
    print(f'optimized accuracy of {np.mean(opt_predict == y_test)} on test data!')

    # save optimized classifier(model) to a file so we can save trained model (eliminating the need
    # for retraining)
    dump(optimized_clf, 'rf_classifier.joblib')

    return optimized_clf


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['train_and_save_classifier'],
        'max-line-length': 100,
        'extra-imports': ['python_ta.contracts', 'pandas', 'numpy', 'joblib',
                          'sklearn.preprocessing', 'sklearn.model_selection',
                          'sklearn.linear_model'],
        'disable': ['R1705', 'C0200'],
    })
