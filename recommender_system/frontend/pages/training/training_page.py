import streamlit as st
from surprise import dump


def training_page_run(data, data_train):
    from model.surprise.training.model_training import model_search
    from model.surprise.inference.recommendation import load_model
    st.header("Training Dashboard 📚")

    st.write("Performing Model Selections:")
    options = st.multiselect(
        'Choose a sets of model for Model selection procedure:',
        ['SVD', 'SVDpp', 'BaselineOnly', 'NormalPredictor'],
        placeholder="Choose Model/s"
        )
    all_options = st.checkbox("Select all options")

    if all_options:
        options = ['SVD', 'SVDpp', 'BaselineOnly', 'NormalPredictor']
    models = [model for model in options]

    if st.button('Train'):
        st.write('Training Model...')

        with st.spinner(f'Doing Model Selections'):
            model_search(data_train)
        st.success("Training model completed!✨",icon="✅")

    st.write("Performing Model Fine-Tuning:")
    option = st.selectbox(
        'Select a model to be tuned:',
        [model for model in data.iloc[:,0]],
        placeholder="Choose Model"
    )
    if st.button('Fine-Tuned'):
        from model.surprise.training.model_training import fine_tuned_model


        st.write(f'Fine Tuning {option} Model...')

        with st.spinner(f'Training {option}... please wait!'):
            if option == 'SVD':
                from surprise import SVD
                fine_tined_model = fine_tuned_model(
                    data_train= data_train,
                    model = option,
                    cv = 5)

                algo = SVD(**fine_tined_model.best_params['rmse'])
                dump.dump("../recommender_system/backend/model/model.pkl", algo=algo)
                st.write(f"SVD MODEL: {fine_tined_model.best_params['rmse']}")

            elif option == 'SVDpp':
                from surprise import SVDpp
                fine_tined_model = fine_tuned_model(
                    data_train= data_train,
                    model = option,
                    cv = 5)

                algo = SVDpp(**fine_tined_model.best_params['rmse'])
                dump.dump("../recommender_system/backend/model/model.pkl", algo=algo)
                st.write(f"SVD MODEL: {fine_tined_model.best_params['rmse']}")

            elif option == 'BaselineOnly':
                from surprise import BaselineOnly
                algo = BaselineOnly()
                dump.dump("../recommender_system/backend/model/model.pkl", algo=algo)

            else :
                from surprise import NormalPredictor
                algo = NormalPredictor()
                dump.dump("../recommender_system/backend/model/model.pkl", algo=algo)

        st.success(f"Fine-Tuning {option} model completed!✨",icon="✅")
