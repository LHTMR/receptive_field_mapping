import streamlit as st


if "y" == 'y':
    from src.components import sidebar
    sidebar.show_sidebar()

st.title('receptive_field_mapping_app')
st.write('This is a streamlit app for interfacing with DeepLabCut in a simple way to make predictions on a video. Followed up by some post-processing to merge it with neuron data.')