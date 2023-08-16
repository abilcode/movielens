import numpy as np
import pandas as pd

def read_data(kind,path=None):
    '''
    This Function is for reading the processing that are desired
    :param kind: which processing wanted to be read (1m/100k)
    :return: ratings, users, movies
    '''
    if kind     =='1m':
        PATH    = "../data/ml-1m"
        if path!= None:
            PATH = f"../{path}/processing/ml-1m"
        rating  = pd.read_csv(f"{PATH}/ratings.dat",
                              sep="::",
                              header=None,
                              names=["MovieID","Title","Rating","Timestamp"],
                              engine='python')

        users   = pd.read_csv(f"{PATH}/users.dat",
                              sep="::",
                              header=None,
                              names=["UserID","Gender","Age","Occupation","Zip-code"],
                              engine='python')

        movies  = pd.read_csv(f"{PATH}/movies.dat",
                              sep="::",
                              header=None,
                              names=["MovieID","Title","Genres"],
                              engine='python',
                              encoding='latin1')
        return rating, users, movies

    else:
        print("Enter the correct Value")
        return None, None, None


