from CONFIG.helper.config import load_config
from model.processing.data import load_data
from model.processing.data import reader_data
from model.suprise.training.model_training import model_search, fine_tuned_model, model_train

model_config = load_config('model.yaml')

rating_data = load_data(
    dataset = 'ml-1m',
    data    = 'ratings.dat',
    cols    = ["UserID","MovieID","Rating"],
)

print(rating_data)

rating_data = reader_data(
    data    = rating_data,
    model   ='surprise',
    cols    = ["UserID","MovieID","Rating"],
    scale   = True
)

# result = model_search(
#
#     data=rating_data,
#     #selected_model = model_config['model']['surprise'],
#     cv = model_config['cv'],
#     metrics = model_config['metrics']
# )
# print(result)
#
# result.to_csv(
#     "../dashboard/backend/data/result_surprise.csv",
#     index=False)
#
model = 'SVD'
params = fine_tuned_model(rating_data, model, cv = 5)

from surprise import SVD
from surprise import dump

algo = SVD(**params.best_params['rmse'])
dump.dump("../analysis/model/model.pkl", algo=algo)