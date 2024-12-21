import streamlit as st
import pandas as pd
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
import dotenv
import os
from io import BytesIO
import gspread
from google.oauth2.service_account import Credentials
from gspread_pandas import Spread

# Load environment variables
dotenv.load_dotenv()
groq_api_key = os.getenv('groq_api_key')

# Google Sheets Configuration
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1vC8rWY1-y7MPvJNzPemww_6-r1uRnabxLVW5soupNJw'

def init_google_sheets():
    """Initialize Google Sheets client."""
    try:
        # Get credentials from .env file
        credentials_data = {
            "type": os.getenv("TYPE"),
            "project_id": os.getenv("PROJECT_ID"),
            "private_key_id": os.getenv("PRIVATE_KEY_ID"),
            "private_key": os.getenv("PRIVATE_KEY").replace('\\n', '\n'),  # Handle newline in private key
            "client_email": os.getenv("CLIENT_EMAIL"),
            "client_id": os.getenv("CLIENT_ID"),
            "auth_uri": os.getenv("AUTH_URI"),
            "token_uri": os.getenv("TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
        }
        
        # Create credentials object
        credentials = Credentials.from_service_account_info(credentials_data, scopes=["https://www.googleapis.com/auth/spreadsheets"])

        # Authorize and return gspread client
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"Error initializing Google Sheets: {str(e)}")
        return None
    
    

class EmailId(BaseModel):
    Company_name: str = Field(..., description="Name of the company")
    Email: str = Field(..., description="Email of the company")
    
class EmailIdResponse(BaseModel):
    data: List[EmailId] = Field(..., description="List of companies and their email addresses")

# Define the prompt template
TEMPLATE = """
Extract company names and their corresponding email addresses from the given text from a linkedin post, following these rules:

Important Rules:
1. ONLY include companies that have an explicitly mentioned email address in the text
2. Skip any company that doesn't have an associated email address
3. Each entry must have both a company name AND a valid email address
4. Do not infer or generate email addresses - only use ones explicitly present in the text
5. Ignore any content present in the comments section

Guidelines for extraction:
- Only extract real email addresses in the format: username@domain.com present in the main text only, ignore comment section
- Maintain exact spelling and formatting of company names and emails
- If a company is mentioned multiple times but has no email, exclude it
- If an email is found without a clear company name, skip it
- Strictly ignore the content present in comments section. Only consider the main text
- Never include colleges, universities or personal email addresses. Only include companies and their email addresses

Text to analyze:
{text}

Return the data in a structured format where each entry contains only valid company-email pairs.
Ensure every company in the output has an associated email address.
Ensure that only companies are present in the output, never include colleges or individual personal profiles and their email addresses.
"""

def process_linkedin_url(url: str, llm) -> List[dict]:
    """Process a single LinkedIn URL and return company-email pairs."""
    try:
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {'verify': False}
        docs = loader.load()
        text = docs[0]

        prompt = PromptTemplate(template=TEMPLATE, input_variables=["text"])
        structured_llm = llm.with_structured_output(EmailIdResponse)
        chain = prompt | structured_llm

        response = chain.invoke({"text": str(text.page_content[:4000])})
        return [{"Company_name": item.Company_name, "E_Mail": item.Email} for item in response.data]
    except Exception as e:
        st.error(f"Error processing URL: {url}\nError: {str(e)}")
        return []

def check_email_exists(email: str, main_sheet) -> bool:
    """Check if email exists in main sheet."""
    try:
        # Get all emails from main sheet
        email_col = main_sheet.find("E_Mail").col
        emails = main_sheet.col_values(email_col)[1:]  # Skip header
        return email in emails
    except Exception as e:
        st.error(f"Error checking email existence: {str(e)}")
        return False

def update_sheet(data: List[dict], selected_group: str, gs_client):
    """Update the selected group sheet with new data."""
    try:
        # Open the spreadsheet using the URL
        spreadsheet = gs_client.open_by_url(SPREADSHEET_URL)
        group_sheet = spreadsheet.worksheet(selected_group)
        main_sheet = spreadsheet.worksheet('main')

        # Filter out existing emails
        new_data = []
        for item in data:
            if not check_email_exists(item['E_Mail'], main_sheet):
                new_data.append(item)
                # Add to main sheet
                main_sheet.append_row([item['Company_name'], item['E_Mail']])

        if new_data:
            # Add to group sheet
            for item in new_data:
                group_sheet.append_row([item['Company_name'], item['E_Mail']])
            
            return len(new_data)
        return 0
    except Exception as e:
        st.error(f"Error updating sheet: {str(e)}")
        return 0

def main():
    st.title("LinkedIn Company Email Extractor")
    st.write("Extract company emails and save to Google Sheets")

    # Initialize Google Sheets client
    gs_client = init_google_sheets()
    if not gs_client:
        st.error("Failed to initialize Google Sheets. Please check your credentials.")
        return

    # Initialize LLM
    llm = ChatOpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
    )

    # Group selection dropdown
    group_options = [f"Group_{i}" for i in range(1, 9)]
    selected_group = st.selectbox("Select Group", group_options)

    # Text area for multiple URLs
    urls_input = st.text_area(
        "Enter LinkedIn URLs (one per line)",
        height=150,
        help="Paste LinkedIn URLs, with each URL on a new line"
    )

    if st.button("Extract and Save Emails"):
        if urls_input.strip():
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process URLs
            urls = urls_input.strip().split('\n')
            all_results = []
            
            for i, url in enumerate(urls):
                if url.strip():
                    status_text.text(f"Processing URL {i+1} of {len(urls)}...")
                    results = process_linkedin_url(url.strip(), llm)
                    all_results.extend(results)
                    progress_bar.progress((i + 1) / len(urls))

            if all_results:
                # Update Google Sheets
                num_added = update_sheet(all_results, selected_group, gs_client)
                
                # Create DataFrame for display
                df = pd.DataFrame(all_results)
                
                # Display results
                st.write(f"### Added {num_added} new entries to {selected_group}")
                st.dataframe(df)
                
                # Provide download option
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Companies')
                
                st.download_button(
                    label="Download Excel File",
                    data=output.getvalue(),
                    file_name="company_emails.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No company emails found in the provided URLs.")

            progress_bar.empty()
            status_text.empty()
        else:
            st.warning("Please enter at least one LinkedIn URL.")

if __name__ == "__main__":
    main()