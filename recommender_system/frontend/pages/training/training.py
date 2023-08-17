import os
import pandas as pd
import streamlit as st
from stqdm import  stqdm
import time





def training_page_run(data):
    st.header("Training Dashboard 📚")

    st.write("Performing Model Selections:")
    options = st.multiselect(
        'Choose a sets of model for Model selection procedure:',
        ['SVD', 'BaselineOnly', 'NormalPredictor'],
        placeholder="Choose Model/s"
        )
    all_options = st.checkbox("Select all options")

    if all_options:
        options = ['SVD', 'BaselineOnly', 'NormalPredictor']
    models = [model for model in options]

    if st.button('Train'):
        st.write('Training Model...')
        for _ in stqdm(range(50)):
            time.sleep(0.5)
        st.success("Training model completed!✨",icon="✅")
        result = data
        st.dataframe(data)


    st.write("Performing Model Fine-Tuning:")
    option = st.selectbox(
        'Select a model to be tuned:',
        [model for model in data.iloc[:,0]],
        placeholder="Choose Model"
    )
    if st.button('Fine-Tuned'):
        from model.suprise.training.model_training import train_selected_model
        from model.processing.data import load_data, reader_data
        st.write(f'Fine Tuning {option} Model...')
        rating_data = load_data(
            dataset = 'ml-1m',
            data    = 'ratings.dat',
            cols    = ["UserID","MovieID","Rating"],
        )
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
        best_params = train_selected_model(data_train= rating_data,model = option)
        st.success(f"Fine-Tuning {option} model completed!✨",icon="✅")
