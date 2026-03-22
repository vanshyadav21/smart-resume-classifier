import pandas as pd
import random

# --- Diverse Sector Templates ---
templates = [
    ("DevOps Engineer", "Amazon / Google", "CI/CD pipelines, Docker, Kubernetes, AWS, Terraform, Jenkins, Cloud Automation, Shell Scripting, Ansible."),
    ("Civil Engineer", "L&T / Tata Projects", "AutoCAD, Structural Design, Site Supervision, Bridge Construction, Project Estimation, Quality Control, Concrete Technology."),
    ("Mechanical Engineer", "Mahindra / Tesla", "SolidWorks, CAD/CAM, Thermodynamics, Machine Design, Industrial Robotics, HVAC, Maintenance, Manufacturing."),
    ("Biochemist", "Pfizer / Sun Pharma", "Molecular Biology, Lab Research, DNA Analysis, Enzyme Studies, Clinical Trials, Chemical Reactions, Biochemistry."),
    ("HR Specialist", "Reliance / Microsoft", "Recruitment, Employee Relations, Payroll Management, Policy Development, Onboarding, Conflict Resolution, Human Resources."),
    ("Data Scientist", "Meta / OpenAI", "Machine Learning, Python, SQL, Predictive Modeling, NLP, Data Visualization, Scikit-learn, TensorFlow, Statistics."),
    ("Software Engineer", "IBM / Samsung", "Debugging, Agile Methodology, Git, Unit Testing, Application Support, Microservices, C++, Java, Spring Boot."),
    ("Research Scientist", "ISRO / NASA", "Hypothesis Testing, Physics Research, Laboratory Safety, Satellite Data Analysis, Technical Documentation, Astronomy."),
    ("Marketing Executive", "Coca-Cola / Nike", "Digital Marketing, SEO, Campaign Management, Social Media, Market Research, Brand Strategy, Content Creation."),
    ("Legal Consultant", "Law Firms Inc.", "Contract Drafting, Legal Research, Compliance, Litigation, Intellectual Property, Corporate Law, Documentation.")
]

# 150 Rows Generate Karne ka Logic
final_rows = []
for i in range(150):
    pos, comp, resp = random.choice(templates)
    prefix = random.choice(["Senior ", "Junior ", "Lead ", "Associate ", ""])
    final_rows.append({
        "positions": prefix + pos,
        "professional_company_names": comp,
        "responsibilities": resp
    })

# CSV Save Karein
df = pd.DataFrame(final_rows)
df.to_csv('resume_data.csv', index=False)
print("✅ Success! 'resume_data.csv' 150 rows ke saath generate ho gayi hai.")
