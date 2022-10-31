"""CSC110 Fall 2021 Final Project: tweet_scrapping.py

This file is used to retrieve the content of the tweets from the COVID-19 Twitter Dataset from
OpenICSPR. This information is needed to train the emotion_classifier.

Copyright and Usage Information
======================================================

This file is provided solely for the personal and private use of staff
marking CSC110 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2021 Bryson Lo, Helena Glowacki, Emma Ho, and Chin Chin Jim."""

import csv
import glob
import time
from typing import Union
import tweepy
import pandas as pd


def scrape_and_process_data() -> None:
    """Scrape twitter for the tweets and process the data"""
    # 1. Getting and formatting the raw data
    data = {
        'emotion': emotion_data_format(),
        'kaggle': kaggle_data_format()
    }

    for key in data:

        current_data = data[key]

        # 2. Scraping and saving tweets' text off twitter using tweepy through tweet ids
        call_batches = 50
        calls_per_batch = 25
        call_batch = 0
        while call_batch < call_batches:

            start_index = (calls_per_batch * 100) * call_batch

            scraped = save_tweets_to_csv(dataset_name=key,
                                         data=current_data,
                                         n_calls=calls_per_batch,
                                         start_index=start_index)

            # scraped is always True unless an error of too many request is raised
            if not scraped:
                print("too many calls! will continue scraping in 15 minutes")
                # in this case, the loop will halt for 15 minutes and rerun this iteration
                time.sleep(60 * 15 + 30)
                continue

            print(f"batch {call_batch} complete...")

            call_batch += 1

        # 3. Combining all scraped tweets into a single file
        create_new_combined_csv(
            original_data=current_data,
            dataset_name=key,
            last_file_index=(calls_per_batch * 100) * (call_batches - 1),
            increment=calls_per_batch * 100)

# 1. Getting and formatting the raw data


def emotion_data_format() -> pd.DataFrame:
    """Return dataframe containing the first 150000 tweet ids of the COVID19 twitter dataset
    """

    # load data
    res = load_data('COVID19_twitter_full_dataset.csv', n_rows=150000)
    # remove all duplicates
    res_no_duplicates = res.drop_duplicates(subset=["tweet_id"]).reset_index(drop=True)

    return res_no_duplicates


def kaggle_data_format() -> pd.DataFrame:
    """Combine the kaggle dataset into a single dataset and return dataframe of tweet ids
    """

    # create a dictionary to paths to each csv file
    real_or_fake = {
        'real': [],
        'fake': []
    }

    # iterate through all the csv files in the folder
    for file_path in glob.iglob("kaggle_data/*.csv"):
        # check if the csv file is a tweet dataset but not tweet replies dataset
        if 'tweet' in file_path and 'tweet_replies' not in file_path:
            if 'Real' in file_path:
                real_or_fake['real'].append(file_path)
            elif 'Fake' in file_path:
                real_or_fake['fake'].append(file_path)

    frames = []

    # load each file and add it to frames
    for key in real_or_fake:
        for file_path in real_or_fake[key]:
            data = load_data(file_path)
            data["real_or_fake"] = key

            frames.append(data)

    # create a combined dataframe of all the tweet data
    res = pd.concat(frames)
    # remove duplicates
    res_no_duplicates = res.drop_duplicates(subset=["tweet_id"]).reset_index(drop=True)

    # split dataset into real and fake
    res_fake = res_no_duplicates[res_no_duplicates["real_or_fake"] == "fake"]
    # get the same number of real observations as fake observations for fair training
    res_real = res_no_duplicates[res_no_duplicates["real_or_fake"] == "real"][:len(res_fake.index)]
    # create a new dataframe once again with an equal number of real and fake observations
    half_real_fake_res = pd.concat([res_real, res_fake])

    return half_real_fake_res


def load_data(path: str, n_rows: int = None) -> pd.DataFrame:
    """Returns a dataframe based on input csv_file
    """
    return pd.read_csv(path, verbose=True, nrows=n_rows)


# 2. Scraping and saving tweets' text off twitter using tweepy through tweet ids

BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAANjIWwEAAAAAJxhMQk09iun9S5jWVlCe9f' \
               'qGCDg%3DPobPJsZi90fX6PnfTUoNzUE7iq3omZ0JjrKDhZx7KqICIEGX1X'
CONSUMER_KEY = '8lHzblKJaZcA7w5zEVkGm3jH9'
CONSUMER_SECRET = 'pnMmiFkA58wpD4z7B6Y5mpaTOEWcnD3GXZ5VoTn0D5ZEWYdlEF'
ACCESS_TOKEN = '1470580211443982338-Vj4kSl9dC57dT29pPYCnykS3oEJSLQ'
ACCESS_TOKEN_SECRET = 'GfP7qtfefD67GfZaDq7Dl0LGQ5De52G19WJlHJdEHg9HO'

AUTH = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
AUTH.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

CLIENT = tweepy.Client(bearer_token=BEARER_TOKEN,
                       consumer_key=CONSUMER_KEY,
                       consumer_secret=CONSUMER_SECRET,
                       access_token=ACCESS_TOKEN,
                       access_token_secret=ACCESS_TOKEN_SECRET)


def get_tweets(tweet_ids: list) -> Union[tweepy.client.Response, None]:
    """Return a list of response objects from Twitter api containing the tweet content
    (and whether or not it could be fetched)
    """
    string_tweet_ids = [str(id1) for id1 in tweet_ids]
    try:
        # if request is successful then result is returned
        res = CLIENT.get_tweets(string_tweet_ids)
        return res
    except tweepy.errors.TooManyRequests:
        # if request is unsuccessful because of requesting too much, None is returned
        return None


def save_tweets_to_csv(dataset_name: str, data: pd.DataFrame, n_calls: int, start_index: int) ->\
        bool:
    """Save scraped tweets to a csv file,
    Return True or False based on whether or not the batch size of tweets being scraped is 100
    """
    # turn dataframe column to list
    tweet_ids = data["tweet_id"].tolist()[start_index: start_index + n_calls * 100]
    # create a dictionary where each key is a tweet id
    tweet_texts_dict = {key: None for key in tweet_ids}
    calls = 0

    while calls < n_calls:

        current_ids = tweet_ids[calls * 100: calls * 100 + 100]

        # checks if batch size is 100, if it is less than that, the loop breaks
        if len(current_ids) == 100:
            res = get_tweets(current_ids)
        else:
            break

        # if None is returned from results, the function stops and returns false
        # this signals to the main function that a timeout is needed due to too many requests
        if res is None:
            return False

        # iterate through the returned data and store tweet text to corresponding tweet id
        data_res = res.data
        for tweet in data_res:
            if tweet.id in tweet_texts_dict.keys():
                tweet_texts_dict[tweet.id] = tweet.text

        # iterate through the returned errors and store None for tweet text for the corresponding
        # tweet ids
        error_res = res.errors
        for error in error_res:
            if error['resource_id'] in tweet_texts_dict.keys():
                tweet_texts_dict[error['resource_id']] = None

        calls += 1

    # save this dictionary at its indexed csv file
    path = f'{start_index}{dataset_name}_res.csv'
    dict_write_to_csv(tweet_texts_dict, path)

    return True


def dict_write_to_csv(dict1: dict[int, Union[str, None]], path: str) -> None:
    """This a helper function that iterates through the dictionary and writes to a csv file
    """
    f = open(path, 'w', newline="")

    writer = csv.writer(f)

    for key in dict1.keys():
        row = [str(key), str(dict1[key]).encode(encoding='UTF-8')]
        writer.writerow(row)

# 3. Combining all scraped tweets into a single file
# As there is a restriction on the number of tweets that can be scrapped off twitter in a certain
# time period. We made multiple calls and thus multiple separate csv files of tweet text.
# Below, we combine all of these tweet texts into a single column, then add it to our original
# dataset as a new variable


def create_tweet_text_dataframe(dataset_name: str,
                                last_file_index: int,
                                increment: int) -> pd.DataFrame:
    """Combines each datafile into a single dataframe and returns it
    """
    file_index = 0

    variable_names = ['tweet_id_2', 'tweet_content']

    frames = []

    # iterate through files and create a dataframe from each csv
    while file_index <= last_file_index:
        print('res_index: ', file_index)

        df = pd.read_csv(f'{file_index}{dataset_name}_res.csv', names=variable_names)
        frames.append(df)

        file_index += increment

    # return concatenated dataframes
    return pd.concat(frames)


def create_new_combined_csv(original_data: pd.DataFrame, dataset_name: str, last_file_index: int,
                            increment: int) -> None:
    """Save combined dataframe to a new file
    """

    tweet_texts = create_tweet_text_dataframe(dataset_name, last_file_index, increment)

    # remove b flag before text in each observation due to csv writing previously
    original_data["tweet_content"] = [text[1:] for text in tweet_texts["tweet_content"].to_list()]

    if 'Unnamed: 0' in original_data.columns:
        original_data.drop(columns='Unnamed: 0')

    if 'index' in original_data.columns:
        original_data.drop(columns='index')

    original_data.to_csv(f'{dataset_name}.csv')


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['scrape_and_process_data',
                       'dict_write_to_csv',
                       'create_tweet_text_dataframe',
                       'get_tweets'],
        'max-line-length': 130,
        'extra-imports': ['python_ta.contracts',
                          'pandas',
                          'tweepy',
                          'csv',
                          'glob',
                          'time',
                          'sys'],
        'disable': ['R1705', 'C0200', 'E9997'],
    })
