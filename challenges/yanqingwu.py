import numpy as np
import pandas as pd
from random import randint

from pathlib import Path
import pickle
import math
# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME: Yanqing Wu
SURNAME: Wu
STUDENT ID: 5142571
KAGGLE ID: Yanqing Wu (Y.Wu-35@student.tudelft.nl)


### NOTES
This files is an example of what your code should look like. 
Any submission in Python 3.* or Java is accepted.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])


def dataframe_info(df):
    print("================================")
    print("\ndf's size = {}, df's shape = {}".format(df.size, df.shape))
    print("\nExample rows...")
    print(df.head(10))
    print(df.tail(10))
    print("\ndescribe df...")
    print(df.describe())
    print("\ndf's info...")
    print(df.info())


# dataframe_info(users_description)
# dataframe_info(movies_description)
# dataframe_info(ratings_description)

''' 
# First observations
1. anormaly data
    - such as age=1 user, may need to treat it differently
2. gender
    - how to encode gender into number?
3. recall from lec that latest movies have higher ratings
    - also recall that item-item CF is more reliable than user-user CF
4. if each user rated each movie, then we should have 6040 * 3706 = 22384240 ratings
    - we only have 910190 ratings
    - very sparse

# Ideas
1. CF+Latent Model
    - different users have different standards, need to normalize star ratings
2. Dealing with missing entries
3. Dealing with anomaly entries
4. process movie names
    - such as use TFIDF
    - maybe add a important-word-score in the ratings-df
5. Dealing with cold-start problem
    - for existing user, use content-based approach
6. Try different N, find albow one?
7. write own RSME evaluation

# ???
    #   2. negative sim score
    #       - https://stats.stackexchange.com/questions/198810/interpreting-negative-cosine-similarity
'''

# overall mean movie rating
global_mu = np.mean(ratings_description['rating'])

# rating deviation of user x
def calc_user_rating_deviation():
    unique_users = np.array(users_description['userID'].unique())
    user_avg_rating = []
    for user in unique_users:
        # print("{} / {}".format(user, len(unique_users)))
        user_entries = ratings_description.loc[ratings_description['userID'] == user]
        user_avg_rating.append(np.mean(user_entries['rating']))

    user_avg_rating = np.array(user_avg_rating)
    _bx = pd.DataFrame(columns=['userID', 'normRating'])
    _bx['userID'] = unique_users
    _bx['normRating'] = user_avg_rating - global_mu

    _bx.to_pickle("./data/bx.pkl")


def calc_movie_rating_deviation():
    unique_movies = np.array(movies_description['movieID'].unique())
    movie_avg_rating = []
    for movie in unique_movies:
        # print("{} / {}".format(movie, len(unique_movies)))
        movie_entries = ratings_description.loc[ratings_description['movieID'] == movie]
        print(movie_entries)
        movie_avg_rating.append(np.mean(movie_entries['rating']))

    movie_avg_rating = np.array(movie_avg_rating)

    _row_mean = pd.DataFrame(columns=['movieID', 'rowMean'])
    _row_mean['movieID'] = unique_movies
    _row_mean['rowMean'] = movie_avg_rating
    _row_mean.to_pickle("./data/row_mean.pkl")

    _bi = pd.DataFrame(columns=['movieID', 'normRating'])
    _bi['movieID'] = unique_movies
    _bi['normRating'] = movie_avg_rating - global_mu
    _bi.to_pickle("./data/bi.pkl")


def construct_user_movie_matrix():
    print("constructing user-movie matrix...")
    num_user = users_description.shape[0]
    num_movie = movies_description.shape[0]
    um_mat = np.zeros(shape=(num_movie+1, num_user+1))
    for _, row in ratings_description.iterrows():
        print(row)
        um_mat[row['movieID']][row['userID']] = row['rating']
    _df = pd.DataFrame(data=um_mat)
    _df.to_csv("./data/user_movie_matrix.csv", index=False)


def preprocess_user_movie_matrix():
    # subtract row mean
    print("preprocessing user-movie matrix...")
    num_user = users_description.shape[0]
    num_movie = movies_description.shape[0]
    um_mat = np.zeros(shape=(num_movie+1, num_user+1))
    for _, row in ratings_description.iterrows():
        # print(row)
        um_mat[row['movieID']][row['userID']] = row['rating'] - row_mean['rowMean'][row['movieID']-1]
        print("{}, {} = {}".format(row['movieID'], row['userID'], um_mat[row['movieID']][row['userID']]))
    _df = pd.DataFrame(data=um_mat)
    _df.to_csv("./data/preprocess_user_movie_matrix.csv", index=False)


def calc_similarity_score():
    # 0. construct user-movie matrix, fill blank as 0
    um_df = None
    if Path("./data/user_movie_matrix.csv").is_file():
        um_df = pd.read_csv("./data/user_movie_matrix.csv")
    else:
        construct_user_movie_matrix()
        um_df = pd.read_csv("./data/user_movie_matrix.csv")
    um_df = um_df.drop(0, axis=0)
    um_df = um_df.drop(['0'], axis=1)
    # dataframe_info(um_df)

    # centered user-movie matrix
    pum_df = None
    if Path("./data/preprocess_user_movie_matrix.csv").is_file():
        pum_df = pd.read_csv("./data/preprocess_user_movie_matrix.csv")
    else:
        preprocess_user_movie_matrix()
        pum_df = pd.read_csv("./data/preprocess_user_movie_matrix.csv")
    pum_df = pum_df.drop(0, axis=0)
    pum_df = pum_df.drop(['0'], axis=1)
    # dataframe_info(pum_df)


    # 3706 3700 -0.0034691579040313308
    # print(np.sum(pum_df.loc[3706]))
    # print(pum_df.loc[3700])

    # 1. subtract mean rating mi from each movie i
    # '''
    num_movie = movies_description.shape[0]
    temp_mat = np.zeros(shape=(num_movie+1, num_movie+1))
    for idx_i, i in pum_df.iterrows():
        for idx_j, j in pum_df.iterrows():
            if (idx_j <= idx_i):
                continue

            ti = i.values
            tj = j.values
            denominator = np.linalg.norm(ti) * np.linalg.norm(tj)

            if denominator is 0:
                score = 0
            else:
                numerator = np.dot(ti, tj)
                score = numerator / denominator
            if math.isnan(score):
                score = 0
            temp_mat[idx_i][idx_j] = score

            print(idx_i, idx_j, score)
    tmp_df = pd.DataFrame(data=temp_mat)
    tmp_df.to_csv("./data/sim_score.csv", index=False)
    # '''

    # 2. compute cosine sim between rows
    return NotImplementedError


# calc_user_rating_deviation()
# calc_movie_rating_deviation()
bx = pd.read_pickle("./data/bx.pkl")
bi = pd.read_pickle("./data/bi.pkl")
row_mean = pd.read_pickle("./data/row_mean.pkl")
# dataframe_info(bx)
# dataframe_info(bi)
# dataframe_info(row_mean)
calc_similarity_score()


def predict(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

#####
##
## SAVE RESULTS
##
#####    


'''
## //!!\\ TO CHANGE by your prediction function
predictions = predict(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
    submission_writer.write(predictions)
'''

# End of File