import os
import pandas as pd
import streamlit as st
import plotly.express as px


def performance_page_run(data):

    result = data
    st.header("Performance dashboard 📊")
    st.write("Model Performance:")
    st.dataframe(result)
    accuracy = result.columns[1]
    fig_col1, fig_col2, fig_col3 = st.columns(3)
    with fig_col1:
        # Create a bar plot using Plotly Express : x = "0" for models
        fig = px.bar(result, x='0', y=result.loc[:,f'{accuracy}'],color='0', title='Accuracy')
        st.plotly_chart(fig, use_container_width=True)

    with fig_col2:
        # Create a bar plot using Plotly Express : x = "0" for models
        fig = px.bar(result, x='0', y=result.loc[:,'test_time'],color='0', title='Test time in second')
        st.plotly_chart(fig, use_container_width=True)

    with fig_col3:
        # Create a bar plot using Plotly Express : x = "0" for models
        fig = px.bar(result, x='0', y=result.loc[:,'fit_time'],color='0', title='Fit Time in second')
        st.plotly_chart(fig, use_container_width=True)

