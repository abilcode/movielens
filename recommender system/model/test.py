import pandas as pd
from CONFIG.helper.config import load_config
from model.processing.data import reader_data
from model.suprise.training.model import model_search

model_config = load_config('model.yaml')

rating_data = pd.read_json(
    "../data/dicoding-data/rating-docding.json"
)
modul_rating = rating_data.iloc[:,[0,2,3]]

print(modul_rating.info())

rating_data = reader_data(
    data    = rating_data,
    model   = 'surprise',
    cols    = ['user_id','module_id','rating'],
    scale   = True
)

result = model_search(
    data    = rating_data,
    cv      = model_config['cv'],
    metrics = model_config['metrics']
)

result.to_csv(
    "../dashboard/backend/data/result_surprise.csv",
    index_label=False,
    index=False
)
