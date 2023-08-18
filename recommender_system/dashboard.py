import streamlit as st
import pandas as pd

from frontend.pages.performance.performance import performance_page_run
from frontend.pages.training.training import training_page_run


st.set_page_config(
    layout="wide",
    page_title="Recommender System Dashboard"
    # This will make the content occupy the full width of the screen
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
        training_page_run(data)

    elif add_radio == 'Model Demo':
        pass

    else :

        st.header("Main dashboard ✨")



if __name__ == "__main__":
    main()


