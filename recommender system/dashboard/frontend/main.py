import streamlit as st
import pandas as pd

from pages.performance.performance import performance_page_run
from pages.training.training import training_page_run


st.set_page_config(
    layout="wide",  # This will make the content occupy the full width of the screen
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
        data = pd.read_csv("../backend/data/result_surprise.csv")
        performance_page_run(data)

    elif add_radio == 'Training':
        training_page_run()
    else :
        col_width = 400
        st.header("Main dashboard ✨")



if __name__ == "__main__":
    main()


