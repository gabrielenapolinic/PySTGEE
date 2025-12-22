import streamlit as st
from google.oauth2 import service_account

# --- Page Configuration ---
st.set_page_config(page_title="GCP Connection Test", layout="centered")

def get_gcp_credentials():
    """
    Retrieves Google Cloud Platform credentials from Streamlit secrets.
    Returns a scoped credentials object.
    """
    # 1. Check if the specific section exists in secrets.toml
    if "gcp_service_account" not in st.secrets:
        st.error("Error: 'gcp_service_account' section not found in secrets.")
        st.stop()

    # 2. Load the secrets dictionary
    # We use dict() to ensure it's a standard dictionary format
    creds_dict = dict(st.secrets["gcp_service_account"])

    # 3. Define Scopes
    # These scopes allow full access to Cloud Platform services.
    # Adjust strictness based on your security needs (e.g., read-only for Sheets).
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    # 4. Create the credentials object
    try:
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=scopes
        )
        return credentials
    except Exception as e:
        st.error(f"Failed to create credentials object: {e}")
        return None

# --- Main Execution ---
def main():
    st.title("🔌 Google Service Account Connection")
    st.markdown("---")

    # Attempt to load credentials
    st.info("Attempting to load credentials from secrets...")
    creds = get_gcp_credentials()

    if creds:
        st.success("✅ Authentication successful!")
        
        # Display safe information to verify the correct account is loaded
        st.subheader("Account Details")
        st.json({
            "project_id": creds.project_id,
            "service_account_email": creds.service_account_email,
            "token_uri": creds.token_uri
        })

        # --- Example: How to use with Google BigQuery (Commented out) ---
        # from google.cloud import bigquery
        # client = bigquery.Client(credentials=creds, project=creds.project_id)
        # query = "SELECT * FROM `my_dataset.my_table` LIMIT 10"
        # df = client.query(query).to_dataframe()
        # st.dataframe(df)

        # --- Example: How to use with Google Sheets (Commented out) ---
        # import gspread
        # gc = gspread.authorize(creds)
        # sh = gc.open("Name of your Google Sheet")
        # worksheet = sh.get_worksheet(0)
        # st.dataframe(worksheet.get_all_records())

    else:
        st.warning("Could not authenticate. Please check your secrets.toml file.")

if __name__ == "__main__":
    main()
