import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME: Yanqing
SURNAME: Wu
STUDENT ID: 5142571
KAGGLE ID: Yanqing Wu


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
    print("\nExample rows...")
    print("\ndf's size = {}, df's shape = {}".format(df.size, df.shape))
    print(df.head(10))
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
'''

# overall mean movie rating
global_mu = np.mean(ratings_description['rating'])

# rating deviation of user x
def calc_rating_deviation():
    unique_users = np.array(users_description['userID'].unique())
    user_avg_rating = []
    _bx = pd.DataFrame(columns=['userID', 'normRating'])
    for user in unique_users:
        # print("{} / {}".format(user, len(unique_users)))
        user_entries = ratings_description.loc[ratings_description['userID'] == user]
        user_avg_rating.append(np.mean(user_entries['rating']))

    user_avg_rating = np.array(user_avg_rating)
    _bx['userID'] = unique_users
    _bx['normRating'] = user_avg_rating - global_mu

    _bx.to_pickle("./data/bx.pkl")


# calc_rating_deviation()
bx = pd.read_pickle("./data/bx.pkl")
print(bx)



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