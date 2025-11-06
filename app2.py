import streamlit as st

# Set the title and page configuration
st.set_page_config(page_title="Hello Diu", layout="centered")

# Main title
st.title("👋 Hello Diu!")

# Subtitle or message
st.subheader("Welcome to your first Streamlit app!")

# Add some interactivity
name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello {name}, greetings from Diu! 🎉")

# A simple image or button can be added too
if st.button("Say Hello"):
    st.balloons()
    st.write("Wishing you a beautiful day from the coastal town of Diu! 🌊🏝️")
