import os


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
    rating = round(prediction.est, 3)
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
    score = []
    for i in range(1,300):
        raitng_user_item = item_rating(model=model,user=2,item=i,print_output=False)['rating']
        score.append((i,raitng_user_item))
    print(score)
    # Sort the list of tuples based on the second element (index 1)
    sorted_score= sorted(score, key=lambda x: x[1], reverse=True)
    print("\n"*2)
    # Print the sorted list of tuples
    print(sorted_score[:20])
