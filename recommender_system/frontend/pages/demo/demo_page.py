import streamlit as st


def demo_page_run():
    st.header("Model Demo: 📚")

    st.subheader("User-Item Inferences:")
    col1, col2 = st.columns(2)

    with col1:
        user_id = st.text_input(
            "Input UserID that want to get recommendations",
            "UserID",
            key="placeholder",
        )
        st.write(f"Choosing used with {user_id} id ")

    with col2:
        num_recs = st.slider(
            "How many recommendations does the User Needs",
            0, 25, 5,
        )
        st.write(f"Outputing {num_recs} recommendations")

    st.subheader("Content or Course Relevancy:")

    st.subheader("User-Item Hybrid Approach:")