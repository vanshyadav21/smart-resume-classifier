📄 PROJECT: AI RESUME INTELLIGENCE PRO (BCA FINAL YEAR)

1. PROJECT OVERVIEW
Yeh ek AI-powered recruitment tool hai jo Resumes (PDFs) ko scan karta hai, unka data extract karta hai, aur Machine Learning (NLP) ka use karke candidate ka "Sector" predict karta hai. Isme ek "Job Recommendation" feature bhi hai jo candidate ke skills ko database se match karta hai.
___________________________________________________________________________________________

2. KEY FEATURES
Deep Resume Analysis: Candidate ka Name, Email, aur Phone number extract karna.

Sector Prediction: NLP keywords ke base par candidate ki industry (DevOps, HR, Civil, etc.) batana.

Job Recommendation: Resume aur Job Description ke beech Cosine Similarity calculate karke best matches dikhana.

OCR Support: Tesseract engine ka use karke scanned/image-based PDFs ko read karna.

Bulk Processing: Ek saath multiple resumes ko process karke CSV report generate karna.

Premium UI: Glassmorphism aur Dark-theme based interactive dashboard.

___________________________________________________________________________________________

3. TECH STACK USED

~ Frontend: Streamlit (Python-based Web Framework)

~ Backend: Python 3.x

~ Machine Learning: Scikit-learn (TfidfVectorizer, RandomForest, Cosine Similarity)

~ NLP & Text Processing: Regex (re), PyPDF2 (Text extraction)

~ OCR Engine: Tesseract OCR, pdf2image

~ Database Management: Pandas

___________________________________________________________________________________________

4. LIBRARIES TO INSTALL
Run this command in your terminal to install all dependencies:

pip install streamlit pandas scikit-learn PyPDF2 pytesseract pdf2image pillow

___________________________________________________________________________________________

5. PROJECT FILES STRUCTURE

app.py - Main application code with UI.

create_csv.py - Script to generate 150+ diverse job entries.

train_model.py - Script to train the AI model and save Pickle files.

resume_data.csv - The job database.

clf.pkl & tfidf.pkl - Saved Machine Learning models.

___________________________________________________________________________________________

6. HOW TO RUN THE PROJECT
Follow these steps in the exact order:

Step 1: Generate Database
Bash
python create_csv.py

Step 2: Train the AI Model
Bash
python train_model.py

Step 3: Launch the Dashboard
ash
python -m streamlit run app.py
___________________________________________________________________________________________

7. FUTURE SCOPE
Integrating LinkedIn API for real-time profile fetching.

Adding Sentiment Analysis for candidate's "Career Objective".

Automated Email notification to shortlisted candidates.