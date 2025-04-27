import streamlit as st
import pickle
import docx
import PyPDF2
import re
import pandas as pd
import numpy as np
from io import StringIO

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
vec = pickle.load(open('vectorizer.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from CSV (new functionality)
def extract_text_from_csv(file):
    # Read CSV using pandas, this assumes the file contains plain text
    text = ''
    df = pd.read_csv(file)
    for column in df.columns:
        text += df[column].to_string(index=False)
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'csv':
        text = extract_text_from_csv(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or CSV file.")
    return text

# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = vec.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# Function to extract contact details (email, phone) from text
def extract_contact_info(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    phone_pattern = r'\+?[0-9]{1,4}?[-.\s\(\)]?\(?[0-9]{1,4}?\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    return emails, phones

# Function to show category details and suggestions
def show_category_details(category):
    if category == "Data Scientist":
        st.info("This role typically requires skills in Python, Machine Learning, Data Visualization, and Statistics.")
    elif category == "Software Engineer":
        st.info("A Software Engineer should highlight programming skills, experience with algorithms, and problem-solving abilities.")
    else:
        st.info(f"Further details about the {category} role can be explored online.")

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, DOCX, TXT, or CSV format and get the predicted job category.")
    
    # Allow multiple file uploads
    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "csv"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Show progress bar
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    resume_text = handle_file_upload(uploaded_file)
                    st.success(f"Successfully extracted text from {uploaded_file.name}.")
                    
                    # Show extracted text (optional)
                    if st.checkbox(f"Show extracted text from {uploaded_file.name}", False):
                        st.text_area("Extracted Resume Text", resume_text, height=300)

                    # Extract and display contact info (email and phone)
                    emails, phones = extract_contact_info(resume_text)
                    if emails:
                        st.write(f"Email(s) found: {', '.join(emails)}")
                    if phones:
                        st.write(f"Phone number(s) found: {', '.join(phones)}")

                    # Make prediction
                    st.subheader(f"Predicted Category for {uploaded_file.name}")
                    category = pred(resume_text)
                    st.write(f"The predicted category of the uploaded resume is: **{category}**")

                    # Show additional category information
                    show_category_details(category)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

if __name__ == "__main__":
    main()
