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

################################
## HELPER FUNCTIONS
################################

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


def read_pickle_file(filepath, method_arg):
    res = None
    if Path(filepath).is_file():
        res = pd.read_pickle(filepath)
    else:
        method_arg()
        res = pd.read_pickle(filepath)
    return res


def read_csv_file(filepath, method_arg):
    df = None
    if Path(filepath).is_file():
        df = pd.read_csv(filepath)
    else:
        method_arg()
        df = pd.read_csv(filepath)
    df = df.drop(0, axis=0)
    df = df.drop(df.columns[0], axis=1)
    return df


def update_ratings_file():
    most_accurate_submission_file = './data/submission_cap5and1.csv'

    r_old = ratings_description
    p = predictions_description
    s = pd.read_csv(most_accurate_submission_file)
    r_new = pd.concat([p, s], axis=1)
    del r_new['Id']
    r_new = r_new.rename(columns={"Rating": "rating"})
    r_new.to_csv('./data/ratings_new.csv', index=False)

    r_comb = pd.concat([r_old, r_new], axis=0)
    r_comb = r_comb.reset_index()
    r_comb.to_csv('./data/ratings_comb.csv', index=False)


################################
## DATA PROCESSING FUNCTIONS
################################


def calc_user_rating_deviation():
    print("Calculating rating deviation of user x...")
    unique_users = np.array(users_description['userID'].unique())
    user_avg_rating = []
    for user in unique_users:
        print("{} / {}".format(user, len(unique_users)))
        user_entries = ratings_description.loc[ratings_description['userID'] == user]
        user_avg_rating.append(np.mean(user_entries['rating']))

    user_avg_rating = np.array(user_avg_rating)
    _bx = pd.DataFrame(columns=['userID', 'normRating'])
    _bx['userID'] = unique_users
    _bx['normRating'] = user_avg_rating - global_mu
    _bx.to_pickle("./data/bx.pkl")


def calc_movie_rating_deviation():
    print("Calculating rating deviation of movie i...")
    unique_movies = np.array(movies_description['movieID'].unique())
    movie_avg_rating = []
    for movie in unique_movies:
        print("{} / {}".format(movie, len(unique_movies)))
        movie_entries = ratings_description.loc[ratings_description['movieID'] == movie]
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
    print("Constructing user-movie matrix...")
    um_mat = np.zeros(shape=(num_movie+1, num_user+1))
    for _, row in ratings_description.iterrows():
        print(row)
        um_mat[row['movieID']][row['userID']] = row['rating']
    _df = pd.DataFrame(data=um_mat)
    _df.to_csv("./data/user_movie_matrix.csv", index=False)


def preprocess_user_movie_matrix():
    # subtract row mean
    print("preprocessing user-movie matrix...")
    um_mat = np.zeros(shape=(num_movie+1, num_user+1))
    for _, row in ratings_description.iterrows():
        um_mat[row['movieID']][row['userID']] = row['rating'] - row_mean['rowMean'][row['movieID']-1]
        print("{}, {} = {}".format(row['movieID'], row['userID'], um_mat[row['movieID']][row['userID']]))
    _df = pd.DataFrame(data=um_mat)
    _df.to_csv("./data/preprocess_user_movie_matrix.csv", index=False)


def calc_similarity_score():
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


def get_exception_rating(x, i):
    # return avg rating for this movie
    movie_i = um_df.iloc[i-1]
    users_rating = movie_i[movie_i > 0]
    users_avg = np.mean(users_rating)
    if users_avg > 0:
        print("[EXCEPTION]: Using users' avg for movie-{}".format(i))
        return users_avg
    else:
        print("[EXCEPTION]: Cold-start for new movie and new user! user-{}, movie-{}".format(x, i))
        return global_mu


def get_rating(x, i, N):
    # x = user_id, i = movie_id
    bx = bx_all.iloc[x-1]['normRating']
    bi = bi_all.iloc[i-1]['normRating']
    bxi = global_mu + bx + bi

    if um_df.iloc[i-1][x-1] != 0:
        # if required prediction has alreaby been rated...
        return um_df.iloc[i-1][x-1]
    else:
        x_movies = um_df.iloc[:, x-1]               # get all movies for user x
        x_watched_movies = x_movies[x_movies > 0]   # filter movies which are watched by user x

    if len(x_watched_movies) is 0:
        print("[WARNING]: USER {} WATCHED 0 MOVIE.".format(x))
        return get_exception_rating(x, i)
    else:
        x_watched_movie_ids = x_watched_movies.index.values

        ver = sim_df.iloc[:,i-1]
        hor = sim_df.iloc[i-1]
        ver = ver[0:i]
        hor = hor[i:num_movie]
        res = np.concatenate((ver.values, hor.values))
        watched_sim = res[x_watched_movie_ids-1]
        idx = np.argpartition(watched_sim, -N)[-N:]
        
        largest_N_scores = watched_sim[idx]
        largest_N_movies = x_watched_movie_ids[idx]
        base_ratings = x_watched_movies.values[idx]
        
        bjs = bj_all[largest_N_movies-1]
        bxjs = global_mu + bx + bjs
        numerator = np.sum(largest_N_scores * (base_ratings - bxjs))
        denominator = np.sum(largest_N_scores)
        if denominator > 0:
            term2 = numerator / denominator
        else:
            print("[WARNING]: Dividing by 0. Replace with exception ratings...")
            return get_exception_rating(x, i)
        return bxi + term2


def predict(predictions):
    prediction_result = []
    # index may need to change to [1,90019]
    for index in range(0, len(predictions)):
        x = predictions.iloc[index]['userID']
        i = predictions.iloc[index]['movieID']
        # TODO: experimenting different N
        N = 10
        y = get_rating(x, i, N)
        if y > 5:
            pred = 5.0
        elif y < 1:
            pred = 1.0
        else:
            pred = y
        prediction_result.append([index+1, pred])
        print(index+1, pred)
    return prediction_result


if __name__ == "__main__":

    ################################
    ## DATA IMPORT
    ################################

    # Where data is located
    movies_file = './data/movies.csv'
    users_file = './data/users.csv'
    ratings_file = './data/ratings_comb.csv'    # NOTICE! modified!
    predictions_file = './data/predictions.csv'
    submission_file = './data/submission.csv'

    # Read the data using pandas
    movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
    users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
    ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
    predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])

    ################################
    ## DATA PRE-PROCESSING
    ################################

    global_mu = np.mean(ratings_description['rating'])
    num_movie = movies_description.shape[0]
    num_user = users_description.shape[0]

    bx_all = read_pickle_file("./data/bx.pkl", calc_user_rating_deviation)
    bi_all = read_pickle_file("./data/bi.pkl", calc_movie_rating_deviation)

    bj_all = bi_all['normRating'].values
    row_mean = pd.read_pickle("./data/row_mean.pkl")

    # construct user-movie matrix, fill blank as 0
    um_df = read_csv_file("./data/user_movie_matrix.csv", construct_user_movie_matrix)

    # centered user-movie matrix; for calculating sim_score
    pum_df = read_csv_file("./data/preprocess_user_movie_matrix.csv", preprocess_user_movie_matrix)

    # calculate similarity score using cosine distance
    sim_df = read_csv_file("./data/sim_score.csv", calc_similarity_score)

    ################################
    ## PREDICTION
    ################################

    predictions = predict(predictions_description)

    # Save predictions, should be in the form 'list of tuples' or 'list of lists'
    with open(submission_file, 'w') as submission_writer:
        # Formates data
        predictions = [map(str, row) for row in predictions]
        predictions = [','.join(row) for row in predictions]
        predictions = 'Id,Rating\n'+'\n'.join(predictions)
        
        # Writes it dowmn
        submission_writer.write(predictions)
    print("[SUCCESS]: Done")


# End of File