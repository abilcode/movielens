import streamlit as st
import pandas as pd

from model.processing.data import load_data, reader_data

from frontend.pages.performance.performance import performance_page_run
from frontend.pages.training.training_page import training_page_run

st.set_page_config(
    # This will make the content occupy the full width of the screen
    layout="wide",
    page_title="Recommender System Dashboard"

)
def main():

    # Using "with" notation
    with st.sidebar:
        add_radio = st.radio(
            "Page Navigation",
            ("Main",
             "Training",
             "Performance",
             "Model Demo")
        )

    if add_radio == 'Performance':
        data = pd.read_csv("backend/data/result_surprise.csv",
                           index_col=False)
        performance_page_run(data)

    elif add_radio == 'Training':
        data = pd.read_csv("backend/data/result_surprise.csv",
                           index_col=False)
        rating_data = load_data(
            dataset = 'ml-1m',
            data    = 'ratings.dat',
            cols    = ["UserID","MovieID","Rating"],
        )

        rating_data = reader_data(
            data    = rating_data,
            model   ='surprise',
            cols    = ["UserID","MovieID","Rating"],
            scale   = True
        )
        training_page_run(data = data,
                          data_train=rating_data)

    elif add_radio == 'Model Demo':
        pass

    else :
        st.header("Main dashboard ✨")

if __name__ == "__main__":
    main()


