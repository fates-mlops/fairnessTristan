import streamlit as st

def app():
    st.title("New Experiment")

    name = st.text_input("Name")

    number = st.number_input("Number", min_value=0, max_value=10, step=1)
    selection = st.selectbox("Selection", ["A", "B", "C"])
    check = st.checkbox("check")

    if st.button("Send"):
        st.write(f"Name: {name}")
        st.write(f"Number: {number}")
        st.write(f"Genre: {selection}")
        st.write(f"Check: {'Yes' if check else 'No'}")