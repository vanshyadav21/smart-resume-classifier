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

# --- 1. CONFIGURATION & OCR SETUP ---
# Standard Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Page Config
st.set_page_config(page_title="Pro RecruitAI Panel | Intelligence Dashboard", page_icon="🎯", layout="wide")

# --- CUSTOM CSS FOR PREMIUM TECH UI ---
st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        background: -webkit-linear-gradient(#3b82f6, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.5);
    }

    /* Sector Prediction Box */
    .sector-box {
        background: rgba(59, 130, 246, 0.15);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #3b82f6;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin: 20px 0;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #3b82f6, #2dd4bf);
        color: white;
        border: none;
        padding: 10px;
        font-weight: bold;
        transition: 0.4s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE LOGIC FUNCTIONS ---

def clean_text(text):
    """Pre-processes text for Vectorization"""
    text = str(text).lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return " ".join(text.split())

def extract_text_with_ocr(file_bytes, page_num=None):
    """OCR Engine for Scanned Documents"""
    try:
        if page_num is not None:
            images = pdf2image.convert_from_bytes(file_bytes, first_page=page_num+1, last_page=page_num+1)
        else:
            images = pdf2image.convert_from_bytes(file_bytes)
        return "".join([pytesseract.image_to_string(img) for img in images])
    except:
        return ""

def get_industry_category(text):
    """Predicts Sector based on Resume Keywords"""
    text = text.lower()
    keywords = {
        "Cloud & DevOps Engineering ☁️": ['docker', 'kubernetes', 'aws', 'devops', 'jenkins', 'terraform', 'ansible'],
        "Software Engineering / IT 💻": ['java', 'react', 'javascript', 'software', 'spring boot', 'frontend', 'backend'],
        "Mechanical Engineering ⚙️": ['mechanical', 'solidworks', 'cad', 'thermodynamics', 'hvac', 'robotics'],
        "Civil Engineering 🏗️": ['civil', 'autocad', 'construction', 'structural', 'site supervisor', 'bridge'],
        "Data Science & AI 🤖": ['python', 'machine learning', 'data science', 'ai', 'tensorflow', 'pandas'],
        "HR & Management 👔": ['hr', 'recruitment', 'payroll', 'onboarding', 'human resource', 'policy'],
        "Science & Bio-Chemistry 🧪": ['biochemist', 'molecular', 'lab', 'biology', 'clinical', 'enzyme', 'chemical']
    }
    for category, keys in keywords.items():
        if any(k in text for k in keys):
            return category
    return "General Professional Services ✨"

def extract_contact(text):
    """Regex for Contact Details"""
    email = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 2]
    name = re.sub(r'[^a-zA-Z\s]', '', lines[0]) if lines else "Unknown"
    return {"Name": name, "Email": email[0] if email else "N/A", "Phone": phone[0] if phone else "N/A"}

# --- 3. ASSET MANAGEMENT ---

@st.cache_resource
def load_resources():
    """Loads CSV and Saved ML Models"""
    try:
        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        df_jobs = pd.read_csv('resume_data.csv')
        # Handle hidden characters in headers
        df_jobs.columns = df_jobs.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        return clf, tfidf, df_jobs
    except:
        return None, None, None

clf, tfidf, df_jobs = load_resources()

# --- 4. NAVIGATION ---

st.sidebar.markdown("<h2 style='text-align: center; color: #3b82f6;'>RecruitAI Pro</h2>", unsafe_allow_html=True)
st.sidebar.divider()
choice = st.sidebar.selectbox("🚀 Control Center", ["Home Dashboard", "Smart Resume Analyzer", "Bulk Talent Sorter"])

if choice == "Home Dashboard":
    st.markdown("<h1 class='main-title'>AI Resume Intelligence Classifier </h1>", unsafe_allow_html=True)
    st.write("<div style='text-align: center; color: #94a3b8;'>Automation and Intelligence for the Modern Recruiter</div>", unsafe_allow_html=True)

elif choice == "Smart Resume Analyzer":
    st.markdown("<h1 class='main-title'>🔍 Candidate Deep-Dive</h1>", unsafe_allow_html=True)
    up_file = st.file_uploader("Upload Candidate Resume (PDF)", type=['pdf'])
    
    if up_file:
        with st.status("Performing AI Analysis...", expanded=True) as status:
            fb = up_file.getvalue()
            pdf = PyPDF2.PdfReader(io.BytesIO(fb))
            raw_text = "".join([p.extract_text() or "" for p in pdf.pages])
            if len(raw_text.strip()) < 50: raw_text = extract_text_with_ocr(fb)
            
            info = extract_contact(raw_text)
            sector = get_industry_category(raw_text)
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # --- Results Grid ---
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Candidate Name", info["Name"])
            with c2: st.metric("Email Address", info["Email"])
            with c3: st.metric("Phone Number", info["Phone"])
            
            st.divider()
            
            # --- Predicted Sector (Above Recommendations) ---
            st.markdown(f"<div class='sector-box'>AI Sector Prediction: {sector}</div>", unsafe_allow_html=True)
            
            # --- Recommendation Logic ---
            if df_jobs is not None:
                st.subheader("🎯 Intelligence-Based Job Matches")
                res_vec = tfidf.transform([clean_text(raw_text)])
                # Comparing against 'responsibilities' column
                job_vecs = tfidf.transform(df_jobs['responsibilities'].astype(str).fillna(''))
                
                scores = cosine_similarity(res_vec, job_vecs)
                top_hits = scores[0].argsort()[-3:][::-1]
                
                rec_df = df_jobs.iloc[top_hits][['positions', 'professional_company_names']]
                rec_df.columns = ["Designation", "Company Name"]
                st.table(rec_df)

else:
    st.markdown("<h1 class='main-title'>📦 Bulk Resume Sorter</h1>", unsafe_allow_html=True)
    b_files = st.file_uploader("Upload Batch PDF Resumes", type=['pdf'], accept_multiple_files=True)
    if b_files:
        rows = []
        for f in b_files:
            fb = f.getvalue()
            reader = PyPDF2.PdfReader(io.BytesIO(fb))
            for i in range(len(reader.pages)):
                t = reader.pages[i].extract_text() or ""
                if len(t.strip()) < 50: t = extract_text_with_ocr(fb, page_num=i)
                det = extract_contact(t)
                cat = get_industry_category(t)
                rows.append({"Candidate": det["Name"], "Sector": cat, "Email": det["Email"], "Phone": det["Phone"]})
        
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button("📤 Export Recruitment Report (CSV)", pd.DataFrame(rows).to_csv(index=False), "Talent_Report.csv")