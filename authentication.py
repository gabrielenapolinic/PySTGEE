import streamlit as st
import ee
import sys
import subprocess
import os

def check_gee_auth():
    """
    Checks GEE authentication. 
    Returns True if connected.
    Returns False (and shows Login UI) if not connected.
    """
    # 1. Try silent connection
    try:
        ee.Initialize(project='stgee-dataset')
        return True
    except:
        pass # Continue to UI

    # 2. Show Login Interface
    st.warning("⚠️ Google Earth Engine access not detected.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔐 Authenticate with Google", type="primary"):
            try:
                # Open browser for authentication
                cmd = "earthengine authenticate"
                if sys.platform == "win32":
                    os.system(f"start cmd /k {cmd}")
                else:
                    subprocess.Popen(cmd.split())
                st.info("Browser opened. Please log in.")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.write("1. Click the button.")
        st.write("2. Login in the browser window.")
        st.write("3. Come back here and click Reload.")
        if st.button("🔄 Reload App"):
            st.rerun()
            
    return False
