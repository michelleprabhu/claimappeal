import streamlit as st

st.title("🚀 Debugging Streamlit App")
st.write("If you see this, Streamlit is rendering correctly.")

# Add a test button
if st.button("Click me"):
    st.success("Streamlit is working!")
