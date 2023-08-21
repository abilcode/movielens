from model.suprise.inference.recommendation import *

def user_item_prediction():
    model = load_model(
        model_filename='../backaend/model/model.pkl'
    )



if __name__ == "__main__":
    user_item_prediction()