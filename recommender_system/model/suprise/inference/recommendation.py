import os
import pickle
from pprint import pprint as pp
from surprise import dump


def load_model(model_filename, verbose=False):
    if verbose :
        print (">> Loading dump")
    from surprise import dump
    import os
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    if verbose:
        print (">> Loaded dump")
    return loaded_model

def item_rating(model,user = 0, item = 0, print_output=False):
    uid = str(user)
    iid = str(item)
    loaded_model = model
    prediction = loaded_model.predict(user, item, verbose=True)
    rating = prediction.est
    details = prediction.details
    uid = prediction.uid
    iid = prediction.iid
    true = prediction.r_ui
    ret = {
        'user': user,
        'item': item,
        'rating': rating,
        'details': details,
        'uid': uid,
        'iid': iid,
        'true': true
    }
    if print_output:
        #pp (ret)
        print('\n\n')
    return ret

if __name__ == "__main__":
    model = load_model(
        os.path.join('/home/dicoding/Documents/CODE/dicoding/project/recsys-explore/recommender system/'
                     'analysis',
                     'model',
                     'model.pkl'
                     )
    )
    print(type(model))

    for i in range(1,300):
        a = item_rating(model=model,user=1,item=i,print_output=False)['rating'];
