import streamlit as st
import pandas as pd
import pickle
import re
import PyPDF2
import pytesseract
from PIL import Image
import pdf2image
import io
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# Windows users check this path:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI Resume Intelligence", layout="wide")

# --- Optimized Helper Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return " ".join(text.split())

def extract_text_with_ocr(file_bytes, page_num=None):
    """Scanned images se text nikalne ke liye optimized OCR"""
    try:
        if page_num is not None:
            images = pdf2image.convert_from_bytes(file_bytes, first_page=page_num+1, last_page=page_num+1)
        else:
            images = pdf2image.convert_from_bytes(file_bytes)
        
        full_text = ""
        for img in images:
            full_text += pytesseract.image_to_string(img)
        return full_text
    except:
        return ""

def extract_info(text):
    """Name, Email aur Skills extract karne ka logic """
    email = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    
    skill_db = ['python', 'java', 'aws', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'sql', 'react', 'autocad', 'tally', 'machine learning', 'marketing']
    extracted_skills = [s.title() for s in skill_db if s in text.lower()]
    
    # Name cleaning logic [cite: 1, 4, 11, 31]
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 2]
    name = "Unknown"
    if lines:
        # Punctuation hatane ke liye filter
        name = re.sub(r'[^a-zA-Z\s]', '', lines[0])
    
    return {
        "Name": name if name else "Unknown",
        "Email": email[0] if email else "N/A",
        "Phone": phone[0] if phone else "N/A",
        "Skills": extracted_skills if extracted_skills else ["None"]
    }

def get_industry_category(text):
    """Keywords ke basis par categorization [cite: 2, 5, 9, 12, 15]"""
    text = text.lower()
    if any(x in text for x in ['docker', 'kubernetes', 'aws', 'devops', 'jenkins']):
        return "Cloud & DevOps" [cite: 1, 2, 3]
    elif any(x in text for x in ['java', 'spring', 'software', 'microservices', 'react']):
        return "Software Engineering" [cite: 14, 15, 16, 20]
    elif any(x in text for x in ['civil', 'autocad', 'construction', 'site supervisor']):
        return "Civil Engineering" [cite: 11, 12, 13]
    elif any(x in text for x in ['hr', 'recruitment', 'payroll', 'onboarding']):
        return "HR & Management" [cite: 8, 9, 10]
    elif any(x in text for x in ['data science', 'machine learning', 'statistics']):
        return "Data Science & AI" [cite: 4, 5, 6]
    elif any(x in text for x in ['marketing', 'seo', 'campaign']):
        return "Marketing" [cite: 31, 32, 35]
    return "General Professional"

@st.cache_resource
def load_resources():
    clf = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    df = pd.read_csv('resume_data.csv')
    return clf, tfidf, df

clf, tfidf, df_jobs = load_resources()

# --- UI Layout ---
st.sidebar.title("🏢 Recruiter Panel")
app_mode = st.sidebar.selectbox("Choose Mode", ["Single Resume Analysis", "Bulk Resume Sorter"])

if app_mode == "Single Resume Analysis":
    st.title("🚀 Smart Resume Classifier")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        with st.spinner("Analyzing..."):
            file_bytes = uploaded_file.getvalue()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            raw_text = "".join([p.extract_text() or "" for p in pdf_reader.pages])
            
            if len(raw_text.strip()) < 50: # OCR required
                raw_text = extract_text_with_ocr(file_bytes)
            
            info = extract_info(raw_text)
            category = get_industry_category(raw_text)
            
            st.header("👤 Candidate Details")
            c1, c2, c3 = st.columns(3)
            c1.metric("Name", info["Name"])
            c2.metric("Email", info["Email"])
            c3.metric("Phone", info["Phone"])
            st.info(f"### 🎯 Category: {category}")
            st.write(f"**Skills:** {', '.join(info['Skills'])}")

else:
    st.title("📂 Bulk Sorter (Multi-Page Splitter)")
    st.write("Is mode mein `Bulk.pdf` ke har page ko alag candidate mana jayega.")
    uploaded_files = st.file_uploader("Upload Resumes", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        results = []
        for file in uploaded_files:
            file_bytes = file.getvalue()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            
            for i in range(len(pdf_reader.pages)):
                text = pdf_reader.pages[i].extract_text() or ""
                if len(text.strip()) < 50: # Image page detected
                    text = extract_text_with_ocr(file_bytes, page_num=i)
                
                info = extract_info(text)
                cat = get_industry_category(text)
                results.append({
                    "Source": f"{file.name} (Pg {i+1})",
                    "Candidate Name": info["Name"],
                    "Category": cat,
                    "Email": info["Email"]
                })
        
        final_df = pd.DataFrame(results)
        st.dataframe(final_df, use_container_width=True)
        st.download_button("📥 Download Excel", final_df.to_csv(index=False), "shortlisted.csv")