import pandas as pd
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("\n--- [1] RE-TRAINING MODEL ---")

csv_file = 'resume_data.csv'
if not os.path.exists(csv_file):
    print(f"❌ ERROR: '{csv_file}' nahi mili!")
else:
    try:
        df = pd.read_csv(csv_file)
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'http\S+\s*', ' ', text)
            text = re.sub(r'[^a-z\s]', ' ', text)
            return " ".join(text.split())

        df['Resume_Text'] = df['skills'].fillna('') + " " + df['responsibilities'].fillna('')
        df['Resume_Text'] = df['Resume_Text'].apply(clean_text)

        tfidf = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2))
        X = tfidf.fit_transform(df['Resume_Text'])
        y = df['job_position_name']

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
        pickle.dump(clf, open('clf.pkl', 'wb'))
        print("⭐⭐ SUCCESS: Model & Vectorizer Saved! ⭐⭐")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")