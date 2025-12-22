import streamlit as st
import PySTGEE  # This imports your logic file

# --- Page Configuration ---
# Must be the first Streamlit command
st.set_page_config(page_title="PySTGEE App", layout="wide")

if __name__ == "__main__":
    try:
        # Calls the main function inside PySTGEE.py
        PySTGEE.run_app()
    except AttributeError:
        st.error("Error: Could not find 'run_app()' in PySTGEE.py. Please ensure you updated the logic file.")
    except Exception as e:
        st.error(f"Critical Error: {e}")
