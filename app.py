import streamlit as st
import pandas as pd
import pickle
import re
import PyPDF2
import pytesseract
import pdf2image
import io
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION & OCR SETUP ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="RecruitAI Pro | Smart ATS", page_icon="🎯", layout="wide")

# --- CUSTOM CSS FOR PREMIUM UI ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    .main-title { font-size: 45px; font-weight: 800; background: -webkit-linear-gradient(#3b82f6, #2dd4bf);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px); padding: 15px; border-radius: 12px; }
    .status-box { background: rgba(59, 130, 246, 0.1); border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE FUNCTIONS ---

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return " ".join(text.split())

def extract_text_with_ocr(file_bytes, page_num=None):
    try:
        images = pdf2image.convert_from_bytes(file_bytes, first_page=(page_num+1 if page_num else 1))
        return "".join([pytesseract.image_to_string(img) for img in images])
    except: return ""

def get_industry_category(text):
    text = text.lower()
    keywords = {
        "Cloud & DevOps Engineering ☁️": ['docker', 'kubernetes', 'aws', 'devops', 'jenkins', 'terraform', 'ansible'],
        "Software Engineering / IT 💻": ['java', 'react', 'javascript', 'node', 'python', 'backend', 'frontend'],
        "Mechanical Engineering ⚙️": ['mechanical', 'solidworks', 'cad', 'thermodynamics', 'hvac', 'robotics'],
        "Civil Engineering 🏗️": ['civil', 'autocad', 'construction', 'structural', 'site engineer'],
        "Data Science & AI 🤖": ['machine learning', 'data science', 'ai', 'tensorflow', 'nlp'],
        "HR & Management 👔": ['hr', 'recruitment', 'payroll', 'onboarding', 'human resource']
    }
    for category, keys in keywords.items():
        if any(k in text for k in keys): return category
    return "General Professional ✨"

def extract_contact(text):
    email = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 2]
    name = re.sub(r'[^a-zA-Z\s]', '', lines[0]) if lines else "Unknown"
    return {"Name": name, "Email": email[0] if email else "N/A", "Phone": phone[0] if phone else "N/A"}

# --- 3. ASSET LOADING ---

@st.cache_resource
def load_assets():
    try:
        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        df_jobs = pd.read_csv('resume_data.csv')
        df_jobs.columns = df_jobs.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        return clf, tfidf, df_jobs
    except: return None, None, None

clf, tfidf, df_jobs = load_assets()

# --- 4. APP UI & NAVIGATION ---

st.sidebar.markdown("<h2 style='text-align: center; color: #3b82f6;'>RecruitAI Panel</h2>", unsafe_allow_html=True)
st.sidebar.divider()
mode = st.sidebar.selectbox("Choose Operation", ["Dashboard", "Single Resume Analysis", "Bulk Recruitment Panel"])

if mode == "Dashboard":
    st.markdown("<h1 class='main-title'>AI Resume Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.image("https://cdn.dribbble.com/users/1233499/screenshots/3850691/data_analysis.gif", use_container_width=True)

elif mode == "Single Resume Analysis":
    st.markdown("<h1 class='main-title'>🔍 Candidate Deep Analysis</h1>", unsafe_allow_html=True)
    f = st.file_uploader("Upload PDF", type=['pdf'])
    if f:
        with st.status("Analyzing...", expanded=True):
            fb = f.getvalue()
            pdf = PyPDF2.PdfReader(io.BytesIO(fb))
            text = "".join([p.extract_text() or "" for p in pdf.pages])
            if len(text.strip()) < 50: text = extract_text_with_ocr(fb)
            
            info = extract_contact(text)
            sector = get_industry_category(text)
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Candidate", info["Name"])
            with c2: st.metric("Email", info["Email"])
            with c3: st.metric("Phone", info["Phone"])
            
            st.markdown(f"<div class='status-box'><b>AI Prediction:</b> {sector}</div>", unsafe_allow_html=True)
            
            if df_jobs is not None:
                st.subheader("🎯 Job Recommendations")
                res_vec = tfidf.transform([clean_text(text)])
                job_vecs = tfidf.transform(df_jobs['responsibilities'].astype(str).fillna(''))
                scores = cosine_similarity(res_vec, job_vecs)
                top = scores[0].argsort()[-3:][::-1]
                st.table(df_jobs.iloc[top][['positions', 'professional_company_names']])

else:
    st.markdown("<h1 class='main-title'>📦 Smart Bulk Talent Sorter</h1>", unsafe_allow_html=True)
    
    # --- RECRUITER CONTROL PANEL ---
    with st.container():
        st.write("### ⚙️ Recruiter Selection Criteria")
        col_a, col_b = st.columns(2)
        with col_a:
            target_role = st.selectbox("I am looking for:", 
                                      ["Cloud & DevOps Engineering ☁️", "Software Engineering / IT 💻", 
                                       "Mechanical Engineering ⚙️", "Civil Engineering 🏗️", 
                                       "Data Science & AI 🤖", "HR & Management 👔"])
        with col_b:
            req_count = st.number_input("Number of candidates needed:", min_value=1, max_value=100, value=10)

    files = st.file_uploader("Upload Batch Resumes", type=['pdf'], accept_multiple_files=True)
    
    if files:
        results = []
        with st.status(f"Scanning {len(files)} resumes...", expanded=True) as s:
            for f in files:
                fb = f.getvalue()
                pdf = PyPDF2.PdfReader(io.BytesIO(fb))
                t = "".join([p.extract_text() or "" for p in pdf.pages])
                if len(t.strip()) < 50: t = extract_text_with_ocr(fb)
                
                info = extract_contact(t)
                cat = get_industry_category(t)
                
                # Get Target Job Responsibilities from CSV
                target_job_name = target_role.split(" ")[0]
                job_row = df_jobs[df_jobs['positions'].str.contains(target_job_name, case=False, na=False)].iloc[0]
                
                # Similarity Calculation (0-100%)
                res_vec = tfidf.transform([clean_text(t)])
                job_vec = tfidf.transform([clean_text(job_row['responsibilities'])])
                score = round(cosine_similarity(res_vec, job_vec)[0][0] * 100, 2)
                
                results.append({
                    "Name": info["Name"], "Email": info["Email"], 
                    "Sector": cat, "Match Score (%)": score, "File": f.name
                })
            s.update(label="Scanning Finished!", state="complete")

        df_res = pd.DataFrame(results)
        
        # --- SMART FILTERING LOGIC ---
        # 1. Sirf wahi dikhao jo selected sector ke hain
        filtered = df_res[df_res['Sector'] == target_role].sort_values(by="Match Score (%)", ascending=False)
        
        st.divider()
        st.subheader(f"🏆 Top {req_count} Recommended Candidates")
        
        if not filtered.empty:
            final_selection = filtered.head(req_count)
            
            # Displaying individual progress bars
            for _, row in final_selection.iterrows():
                c1, c2 = st.columns([4, 1])
                c1.write(f"✅ **{row['Name']}** | {row['Email']}")
                c2.progress(int(row['Match Score (%)']) / 100, text=f"{row['Match Score (%)']}%")
            
            st.write("---")
            st.dataframe(final_selection, use_container_width=True)
            st.download_button("📤 Download Recruitment Report", final_selection.to_csv(index=False), "Shortlist.csv")
        else:
            st.warning("No matching candidates found for this specific role in this batch.")