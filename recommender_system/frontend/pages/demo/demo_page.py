import os
import streamlit as st
from model.processing.data import load_data
from model.surprise.inference.recommendation import load_model, item_rating
from model.processing.similarity import cosine_similarity_matrix

def demo_page_run():
    st.header("Model Demo: 📚")
    model = load_model(
        model_filename=os.path.join('/home/dicoding/Documents/CODE/dicoding/project/recsys-explore/recommender system/'
                                    'analysis',
                                    'model',
                                    'model.pkl'
                                    )
    )
    st.subheader("User-Item Inferences:")
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.text_input(
            "Input UserID that want to get recommendations",
            "UserID",
            key="user id",
        )
        st.write(f"Choosing used with {user_id} id ")

    with col2:
        num_recs = st.slider(
            "How many recommendations does the User Needs",
            0, 25, 5,
        )
        st.write(f"Outputing {num_recs} recommendations")

    score = []
    subset = [i for i in range(300) if i % 2 ==0]
    for i in range(1,300):
        if i not in subset:
            rating_user_item = item_rating(model=model,user=user_id,item=i,print_output=False)['rating']
            score.append((i,rating_user_item))
    # Sort the list of tuples based on the second element (index 1)
    sorted_score = sorted(score, key=lambda x: x[1], reverse=True)
    st.write(f"Here are the {num_recs} recs:{sorted_score[:num_recs]}")

    st.subheader("Content or Course Relevancy:")
    movie_data = load_data(
        dataset = 'ml-1m',
        data    = 'movies.dat',
        cols    = ["MovieID","Title","Genres"],
    )
    #st.dataframe(movie_data[movie_data['MovieID']==1])
    col1, col2 = st.columns(2)
    with col1:
        movie_id = st.text_input(
            "Movie:",
            "MovieID",
            key="movie id text",
        )
    st.write(f"Selecting Movie: {movie_data[movie_data['MovieID']==int(movie_id)].loc[:,'Title']}")
    with col2:
        num_recs = st.slider(
            "How many recommendations does the User Needs",
            0, 25, 5,
            key="movie id slider"
        )
        st.write(f"Outputing {num_recs} recommendations")
    similarity_matrix = cosine_similarity_matrix(movie_data.Genres)
    movie_scores = list(enumerate(similarity_matrix[0]))
    movie_scores = [ i for i in movie_scores if i[1] != 1]
    sorted_score = sorted(movie_scores, key=lambda x: x[1], reverse=True)
    st.write(f"Here are the {num_recs} recs:{sorted_score[:num_recs]}")
    #st.subheader("User-Item Hybrid Approach:")