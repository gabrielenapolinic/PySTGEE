import streamlit as st
import pandas as pd
import gspread
from google.oauth2 import service_account

# --- Page Configuration ---
st.set_page_config(page_title="GSheet Explorer", layout="wide")

def get_gcp_credentials():
    """
    Retrieves Google Cloud Platform credentials from Streamlit secrets.
    Returns a scoped credentials object.
    """
    if "gcp_service_account" not in st.secrets:
        st.error("Error: 'gcp_service_account' section not found in secrets.")
        st.stop()

    creds_dict = dict(st.secrets["gcp_service_account"])

    # Scopes needed for Sheets and Drive
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    try:
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=scopes
        )
        return credentials
    except Exception as e:
        st.error(f"Failed to create credentials object: {e}")
        return None

def load_data(credentials, sheet_name):
    """
    Connects to Google Sheets and retrieves data from the first worksheet.
    """
    try:
        # Authorize the gspread client
        gc = gspread.authorize(credentials)
        
        # Open the spreadsheet by name
        sh = gc.open(sheet_name)
        
        # Get the first worksheet
        worksheet = sh.get_worksheet(0)
        
        # Get all records as a list of dicts
        data = worksheet.get_all_records()
        
        return pd.DataFrame(data)
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"❌ Spreadsheet '{sheet_name}' not found. Did you share it with the service account?")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        return None

# --- Main Execution ---
def main():
    st.title("📊 Google Sheets Data Viewer")
    st.markdown("""
    This app connects to your Google Drive via Service Account.
    **Remember:** You must share your sheet with: 
    `66534163922-compute@developer.gserviceaccount.com`
    """)
    st.markdown("---")

    # 1. Load Credentials
    creds = get_gcp_credentials()

    if creds:
        # 2. Input for Sheet Name
        sheet_name = st.text_input(
            "Enter the exact name of your Google Sheet:", 
            placeholder="e.g., My Test Data"
        )

        if sheet_name:
            with st.spinner(f"Loading data from '{sheet_name}'..."):
                # 3. Fetch Data
                df = load_data(creds, sheet_name)

                if df is not None:
                    st.success("Data loaded successfully!")
                    
                    # 4. Data Visualization
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader("Data Preview")
                        st.dataframe(df, use_container_width=True)
                    
                    with col2:
                        st.subheader("Stats")
                        st.write(f"**Rows:** {df.shape[0]}")
                        st.write(f"**Columns:** {df.shape[1]}")
                        st.write("Column Names:")
                        st.code("\n".join(df.columns))

                    # 5. Simple Chart (if numeric data exists)
                    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
                    if len(numeric_cols) > 0:
                        st.subheader("Quick Chart")
                        chart_col = st.selectbox("Select column to graph:", numeric_cols)
                        st.bar_chart(df[chart_col])

if __name__ == "__main__":
    main()
