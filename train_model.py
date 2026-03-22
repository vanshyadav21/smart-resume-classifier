import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def train_and_save():
    # 1. Check if CSV exists
    if not os.path.exists('resume_data.csv'):
        print("❌ Error: 'resume_data.csv' nahi mili! Pehle 'create_csv.py' run karein.")
        return

    try:
        # 2. Load CSV
        df = pd.read_csv('resume_data.csv')
        
        # Hidden characters hatane ke liye columns ko clean karein
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        print(f"✅ CSV Loaded. Available columns: {list(df.columns)}")

        # 3. Data Prepare Karein (Using your specific columns)
        # 'responsibilities' training text hai aur 'positions' labels hain
        X = df['responsibilities'].astype(str).fillna('General Professional')
        y = df['positions'].astype(str).fillna('General Role')

        # 4. TF-IDF Vectorizer (Text ko numbers mein badalne ke liye)
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X_tfidf = tfidf.fit_transform(X)

        # 5. Model Training (Random Forest)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_tfidf, y)

        # 6. Save Files (Yahi files app.py mang raha hai)
        pickle.dump(clf, open('clf.pkl', 'wb'))
        pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

        print("\n🎉 MUBARAK HO! 'clf.pkl' and 'tfidf.pkl' generate ho gayi hain.")
        print("Ab aap 'streamlit run app.py' chala sakte hain.")

    except Exception as e:
        print(f"❌ Kuch galat hua: {e}")

if __name__ == "__main__":
    train_and_save()