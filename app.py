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

# Load environment variables
dotenv.load_dotenv()
groq_api_key = os.getenv('groq_api_key')

# Define the data models
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
        # Load the webpage
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {'verify': False}
        docs = loader.load()
        text = docs[0]

        # Create the prompt and chain
        prompt = PromptTemplate(template=TEMPLATE, input_variables=["text"])
        structured_llm = llm.with_structured_output(EmailIdResponse)
        chain = prompt | structured_llm

        # Process the text
        response = chain.invoke({"text": str(text.page_content[:4000])})
        return [{"Company Name": item.Company_name, "Email": item.Email} for item in response.data]
    except Exception as e:
        st.error(f"Error processing URL: {url}\nError: {str(e)}")
        return []

def main():
    st.title("LinkedIn Company Email Extractor")
    st.write("Extract company emails from multiple LinkedIn posts")

    # Initialize LLM
    llm = ChatOpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
    )

    # Text area for multiple URLs
    urls_input = st.text_area(
        "Enter LinkedIn URLs (one per line)",
        height=150,
        help="Paste LinkedIn URLs, with each URL on a new line"
    )

    if st.button("Extract Emails"):
        if urls_input.strip():
            # Show progress bar
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
                # Create DataFrame
                df = pd.DataFrame(all_results)
                
                # Create Excel file in memory
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Companies')
                
                # Provide download button
                st.download_button(
                    label="Download Excel File",
                    data=output.getvalue(),
                    file_name="company_emails.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Display results in the app
                st.write("### Extracted Data:")
                st.dataframe(df)
            else:
                st.warning("No company emails found in the provided URLs.")

            # Clear progress bar and status
            progress_bar.empty()
            status_text.empty()
        else:
            st.warning("Please enter at least one LinkedIn URL.")

if __name__ == "__main__":
    main()