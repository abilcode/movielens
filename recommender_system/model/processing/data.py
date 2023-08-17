import os
import chardet
import pandas as pd
from surprise import Dataset, Reader


def load_data(dataset, data = 'ratings.dat', cols = None):
    '''

    :param dataset_name:
    :return:
    '''

    def detect_encoding(file_path):
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']

    data_path = os.path.join(
        os.path.dirname(__file__),
        '../../data',
        dataset,
        data
    )

    data = pd.read_csv(
        data_path,
        sep="::",
        header=None,
        index_col = False,
        names=cols,
        engine='python',
        encoding=detect_encoding(data_path)
    )
    return data

def reader_data(data, model=None, cols = ["UserID", "MovieID", "Rating"], scale = False):
    '''

    :param df:
    :param model:
    :return:
    '''
    cols = cols
    if model == 'surprise':
        if scale:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(data[cols],
                                        reader)

        return data
    if model == 'lightfm':
        items_feature_columns = cols



