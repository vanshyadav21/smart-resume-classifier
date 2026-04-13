import streamlit as st
import pandas as pd
import pickle
import re
import hashlib
import json
import os
import io
import PyPDF2
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIG & DB SETUP ---
DB_FILE = "users_db.json"
st.set_page_config(page_title="RecruitAI Enterprise", page_icon="💎", layout="wide")

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: return json.load(f)
    return {"admin": hashlib.sha256("admin123".encode()).hexdigest()}

def save_to_db(db):
    with open(DB_FILE, "w") as f: json.dump(db, f)

def make_hashes(password): return hashlib.sha256(str.encode(password)).hexdigest()

# Session State Initialization
if 'user_db' not in st.session_state: st.session_state['user_db'] = load_db()
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'auth_mode' not in st.session_state: st.session_state['auth_mode'] = None
if 'page' not in st.session_state: st.session_state['page'] = "Dashboard"
if 'selected_candidate' not in st.session_state: st.session_state['selected_candidate'] = None
if 'history' not in st.session_state: st.session_state['history'] = []

# --- 2. CUSTOM CSS (BLUE-BLACK + MEGA PROFILE UI) ---
st.markdown("""
    <style>
    .stApp { background: #000000; color: #ffffff; }
    h1, h2, h3 { color: #3b82f6 !important; font-family: 'Segoe UI', sans-serif; }
    .main-title { font-size: 50px; font-weight: 800; text-align: center; color: #3b82f6; text-shadow: 0 0 15px rgba(59, 130, 246, 0.4); }
    
    /* Login Centered Box */
    .login-box {
        background: rgba(15, 23, 42, 0.9); border: 2px solid #3b82f6; 
        padding: 40px; border-radius: 15px; text-align: center;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); margin-top: 50px;
    }

    /* Equal Sidebar Buttons */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100% !important; background: transparent !important;
        color: #3b82f6 !important; border: 1px solid #3b82f6 !important;
        margin-bottom: 8px; text-transform: uppercase; font-size: 12px;
        display: block !important;
    }

    .feature-tile, .card {
        background: rgba(15, 23, 42, 0.9); border: 2px solid #3b82f6; 
        padding: 40px; border-radius: 15px; text-align: center;
    }
    .feature-tile:hover { transform: translateY(-5px); box-shadow: 0 0 25px rgba(59, 130, 246, 0.5); transition: 0.3s; }

    /* MEGA PROFILE UI (Large Icons & Text) */
    .mega-header {
        background: rgba(59, 130, 246, 0.15); padding: 40px; 
        border-radius: 25px; border: 2px solid #3b82f6; margin-bottom: 30px;
    }
    .mega-name { font-size: 60px !important; font-weight: 800; margin: 0; color: #ffffff !important; }
    .mega-contact { font-size: 22px; color: #3b82f6; margin-top: 10px; }
    
    .match-circle {
        font-size: 55px; font-weight: 900; color: #3b82f6;
        text-align: center; border: 6px solid #3b82f6;
        border-radius: 50%; width: 180px; height: 180px;
        display: flex; align-items: center; justify-content: center;
        margin: auto; box-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
        background: rgba(0,0,0,0.5);
    }
    
    .stButton>button { 
        background: transparent !important; color: #3b82f6 !important; border: 2px solid #3b82f6 !important; 
        font-weight: bold !important; width: 100% !important; border-radius: 10px !important; height: 3.5em !important;
    }
    .stButton>button:hover { background: #3b82f6 !important; color: #ffffff !important; box-shadow: 0 0 15px #3b82f6; }
    
    [data-testid="stSidebar"] { background-color: #020617 !important; border-right: 1px solid #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return " ".join(text.split())

def extract_info(text):
    email = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 2]
    name = lines[0] if lines else "Unknown"
    skills_list = ['python', 'java', 'aws', 'docker', 'kubernetes', 'react', 'sql', 'jenkins', 'terraform', 'git']
    found = [s.upper() for s in skills_list if s in text.lower()]
    return {"Name": name, "Email": email[0] if email else "N/A", "Phone": phone[0] if phone else "N/A", "Skills": found}

@st.cache_resource
def load_assets():
    try:
        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        df_jobs = pd.read_csv('resume_data.csv')
        return clf, tfidf, df_jobs
    except: return None, None, None

clf, tfidf, df_jobs = load_assets()

# --- 4. SHARED UI COMPONENT: MEGA PROFILE VIEW ---
def display_mega_profile(c_data):
    st.markdown(f"""
    <div class='mega-header'>
        <div style='display: flex; align-items: center;'>
            <div style='font-size: 100px; margin-right: 40px;'>👤</div>
            <div>
                <h1 class='mega-name'>{c_data['Name']}</h1>
                <p class='mega-contact'>📧 {c_data['Email']} | 📞 {c_data['Phone']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"<br><div class='match-circle'>{c_data['Score']}%</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-weight:bold; color:#3b82f6; font-size:20px; margin-top:15px;'>MATCH SCORE</p>", unsafe_allow_html=True)
    with col2:
        st.write("### 🛠️ Professional Skills Found")
        if c_data['Skills']:
            st.write(", ".join([f"`{s}`" for s in c_data['Skills']]))
        else: st.info("General Skills")
        st.write("### 📊 AI Analysis Summary")
        st.write(f"This candidate shows compatibility with the **{c_data['Target']}** role requirements.")
        st.progress(c_data['Score']/100)

# --- 5. PAGE ROUTING & FUNCTIONS ---

def login_ui():
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.write("##")
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.markdown("<h1>RECRUIT-AI</h1><p style='color:#64748b;'>v2.0 ENTERPRISE PORTAL</p><hr>", unsafe_allow_html=True)
        if st.session_state['auth_mode'] is None:
            c1, c2 = st.columns(2)
            if c1.button("LOG IN"): st.session_state['auth_mode'] = 'login'; st.rerun()
            if c2.button("CREATE"): st.session_state['auth_mode'] = 'signup'; st.rerun()
        elif st.session_state['auth_mode'] == 'login':
            u = st.text_input("Username")
            p = st.text_input("Password", type='password')
            if st.button("AUTHENTICATE"):
                if u in st.session_state['user_db'] and make_hashes(p) == st.session_state['user_db'][u]:
                    st.session_state['logged_in'] = True; st.session_state['username'] = u; st.rerun()
                else: st.error("Access Denied!")
            if st.button("⬅ BACK"): st.session_state['auth_mode'] = None; st.rerun()
        elif st.session_state['auth_mode'] == 'signup':
            nu = st.text_input("New Identity")
            np = st.text_input("Security Key", type='password')
            if st.button("REGISTER"):
                if nu and np:
                    st.session_state['user_db'][nu] = make_hashes(np); save_to_db(st.session_state['user_db'])
                    st.success("Account Created. Please Login."); st.session_state['auth_mode'] = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def dashboard_page():
    st.markdown("<h1 class='main-title'>AI Resume Classifier</h1><hr>", unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        st.markdown("<div class='feature-tile'><h1>🔍</h1><h3>SINGLE SCANNER</h3></div>", unsafe_allow_html=True)
        if st.button("OPEN SINGLE SCAN"): st.session_state['page'] = "Single"; st.rerun()
    with r:
        st.markdown("<div class='feature-tile'><h1>📁</h1><h3>BULK SORTER</h3></div>", unsafe_allow_html=True)
        if st.button("OPEN BULK SORTER"): st.session_state['page'] = "Bulk"; st.rerun()
    st.write("##")
    st.area_chart(pd.DataFrame({'Traffic': [10, 40, 30, 80, 50]}), color="#3b82f6")

def single_scanner_page():
    st.markdown("<h1 class='main-title'>Individual Intelligence</h1>", unsafe_allow_html=True)
    if st.button("⬅ Back to Dashboard"): st.session_state['page'] = "Dashboard"; st.rerun()
    
    target = st.selectbox("Job Profile:", ["DevOps Engineer", "Data Scientist", "Java Developer", "Software Engineer"])
    f = st.file_uploader("Upload Resume PDF", type=['pdf'])
    
    if f:
        pdf = PyPDF2.PdfReader(f)
        text = "".join([p.extract_text() or "" for p in pdf.pages])
        info = extract_info(text)
        score = round(cosine_similarity(tfidf.transform([clean_text(text)]), tfidf.transform([clean_text(target)]))[0][0] * 100, 2)
        
        st.divider()
        # --- MEGA UI FOR SINGLE SCANNER ---
        display_mega_profile({**info, "Score": score, "Target": target})
        st.session_state['history'].append({"t": datetime.now().strftime("%H:%M"), "n": f.name})

def bulk_sorter_page():
    st.markdown("<h1 class='main-title'>Smart Bulk Sorter</h1>", unsafe_allow_html=True)
    if st.button("⬅ Dashboard"): st.session_state['page'] = "Dashboard"; st.rerun()
    c1, c2 = st.columns(2)
    target = c1.selectbox("Criteria:", ["DevOps Engineer", "Data Scientist", "Java Developer", "Software Engineer"])
    count = c2.number_input("Candidates Needed:", min_value=1, value=5)
    files = st.file_uploader("Upload Batch Resumes", type=['pdf'], accept_multiple_files=True)
    
    if files:
        if len(files) < count: st.warning(f"⚠️ Requested {count} profiles, but only {len(files)} uploaded.")
        results = []
        for f in files:
            pdf = PyPDF2.PdfReader(f); t = "".join([p.extract_text() or "" for p in pdf.pages])
            info = extract_info(t)
            score = round(cosine_similarity(tfidf.transform([clean_text(t)]), tfidf.transform([clean_text(target)]))[0][0] * 100, 2)
            results.append({**info, "Score": score, "Target": target})
        
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False).head(int(count))
        st.divider()
        for idx, row in df.iterrows():
            cb, ct = st.columns([1, 4])
            if cb.button("VIEW PROFILE", key=f"v_{idx}"):
                st.session_state['selected_candidate'] = row.to_dict(); st.session_state['page'] = "ProfileView"; st.rerun()
            ct.markdown(f"**{row['Name']}** | Match Score: {row['Score']}%")
            st.progress(row['Score']/100)

# --- 6. EXECUTION FLOW ---
if not st.session_state['logged_in']:
    login_ui()
else:
    st.sidebar.markdown(f"<h3 style='text-align: center; color:#3b82f6;'>👤 {st.session_state['username']}</h3>", unsafe_allow_html=True)
    st.sidebar.divider()
    if st.sidebar.button("📊 DASHBOARD"): st.session_state['page'] = "Dashboard"; st.rerun()
    if st.sidebar.button("📜 HISTORY"): st.session_state['page'] = "History"; st.rerun()
    if st.sidebar.button("ℹ️ ABOUT ME"): st.session_state['page'] = "About"; st.rerun()
    if st.sidebar.button("💬 FEEDBACK"): st.session_state['page'] = "Feedback"; st.rerun()
    st.sidebar.write("##")
    if st.sidebar.button("🚪 LOGOUT"): st.session_state['logged_in'] = False; st.session_state['auth_mode'] = None; st.rerun()

    if st.session_state['page'] == "Dashboard": dashboard_page()
    elif st.session_state['page'] == "Single": single_scanner_page()
    elif st.session_state['page'] == "Bulk": bulk_sorter_page()
    elif st.session_state['page'] == "ProfileView":
        if st.button("⬅ Back to Results"): st.session_state['page'] = "Bulk"; st.rerun()
        display_mega_profile(st.session_state['selected_candidate'])
    elif st.session_state['page'] == "History":
        st.markdown("<h1>📜 History</h1>", unsafe_allow_html=True)
        for h in st.session_state['history']: st.write(f"🕒 {h['t']} - {h['n']}")
    elif st.session_state['page'] == "About":
        st.markdown("<div class='card'><h3>Vansh Yadav</h3><p>BCA Final Year Project</p></div>", unsafe_allow_html=True)
    elif st.session_state['page'] == "Feedback":
        st.text_area("Message")
        if st.button("Submit"): st.success("Logged!")