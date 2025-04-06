import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load generation model from Hugging Face
generator = pipeline("text2text-generation", model="google/flan-t5-small")

#shl load generation
data = [
    {"id": 1, "name": "Sales Potential Test", "role": "Sales", "seniority": "Entry", "skills": "Communication, Persuasion"},
    {"id": 2, "name": "Technical Aptitude Test", "role": "Tech", "seniority": "Entry", "skills": "Problem Solving, Logic"},
    {"id": 3, "name": "Managerial Assessment", "role": "Manager", "seniority": "Mid", "skills": "Leadership, Decision Making"},
    {"id": 4, "name": "Coding Simulation", "role": "Tech", "seniority": "Mid", "skills": "Python, Coding, Problem Solving"},
    {"id": 5, "name": "Customer Support Test", "role": "Support", "seniority": "Entry", "skills": "Empathy, Communication"},
    {"id": 6, "name": "Data Analysis Test", "role": "Analytics", "seniority": "Mid", "skills": "Data Interpretation, Excel"},
]
catalog = pd.DataFrame(data)
catalog['text'] = catalog['role'] + " " + catalog['seniority'] + " " + catalog['skills'] + " " + catalog['name']
catalog_embeddings = embedder.encode(catalog['text'].tolist(), convert_to_tensor=True)

# Streamlit Page Config
st.set_page_config(page_title="SHL GenAI Assessment Tool", layout="wide")

# Custom SHL-like Styling
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9f9fb;
        color: #222;
    }
    .main {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    h1, h2, h3, h4 {
        color: #4b2e83;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4b2e83;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3a2367;
    }
    .logo {
        width: 160px;
        margin-bottom: 10px;
    }
    .navbar {
        background-color: #4b2e83;
        padding: 0.8rem 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: white;
    }
    .navbar a {
        color: white;
        margin-right: 1rem;
        text-decoration: none;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="navbar">
    <a href="#">Home</a>
    <a href="#">Solutions</a>
    <a href="#">Products</a>
    <a href="#">HR Priorities</a>
    <a href="#">Resources</a>
    <a href="#">Careers</a>
    <a href="#">About</a>
    <a href="#">Contact</a>
    <a href="#">Practice Tests</a>
    <a href="#">Support</a>
    <a href="#" style="float:right;">Login</a>
    <a href="#" style="float:right;">Buy Online</a>
</div>
""", unsafe_allow_html=True)

# SHL Brand Logo (fallback to header text)
st.markdown("<h1 style='color:#4b2e83;'>SHL</h1>", unsafe_allow_html=True)

# Title and Description
st.markdown("""
<div class="main">
<h1> SHL GenAI Assessment Recommendation Tool</h1>
<p>Welcome to the <strong>SHL GenAI Assessment Recommendation Tool</strong>! </p>
<p>Paste a job description below, and let our AI recommend the best-fit assessments from our catalog based on the role, seniority, and skills required.</p>
<p><strong>Find assessments that best meet your needs.</strong><br>
Browse through our extensive product catalog for science-backed assessments that evaluate cognitive ability, personality, behavior, skills, and more, by role and organizational level, by industry, and by language.</p>
""", unsafe_allow_html=True)

# Input Area
job_input = st.text_area(" Paste the Job Description or Role Info:", height=200)

# Generate Button
if st.button("Generate Recommendations"):
    if job_input.strip() == "":
        st.warning(" Please enter a job description to proceed.")
    else:
        with st.spinner("Thinking with GenAI..."):
            input_embedding = embedder.encode(job_input, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(input_embedding, catalog_embeddings)[0]
            top_k = min(3, len(similarities))
            top_results = similarities.topk(k=top_k)

            retrieved_info = ""
            top_assessments = []
            for score, idx in zip(top_results.values, top_results.indices):
                item = catalog.iloc[idx.item()]
                top_assessments.append(item)
                retrieved_info += f"- {item['name']} ({item['role']}, {item['seniority']}): {item['skills']}\n"

            prompt = f"Job Description: {job_input}\n\nAvailable Assessments:\n{retrieved_info}\n\nBased on the job description and the assessments, which ones are most suitable and why?"
            output = generator(prompt, max_length=256, do_sample=False)[0]['generated_text']

            st.subheader(" AI Recommendations:")
            st.success(output)

            st.markdown("""
---
<h4> Retrieved Assessments:</h4>
""", unsafe_allow_html=True)
            for i, a in enumerate(top_assessments, 1):
                st.markdown(f"**{i}. {a['name']}**<br>• Role: {a['role']}<br>• Seniority: {a['seniority']}<br>• Skills: {a['skills']}", unsafe_allow_html=True)

            # Downloadable report button
            report_md = f"### SHL GenAI Recommendations\n\n**Job Description:**\n{job_input}\n\n**AI Recommendation:**\n{output}\n\n**Top Matches:**\n" + "\n".join([f"{i+1}. {a['name']} ({a['role']}, {a['seniority']}): {a['skills']}" for i, a in enumerate(top_assessments)])
            st.download_button(
                label=" Download Recommendations",
                data=report_md,
                file_name="shl_genai_recommendations.txt",
                mime="text/plain"
            )

# Footer
st.markdown("""
<hr>
<p style="text-align:center;font-size:0.9em;">Made with  using <a href="https://streamlit.io/" target="_blank">Streamlit</a> and <a href="https://huggingface.co/" target="_blank">Hugging Face</a>.</p>
<head>
    <style>
        #footer {
            position: sticky;
            padding: 5px;
            bottom: 0;
            width: 80%;
            height: 50px;
            background: white(55);
            color: white;
            font-size: 30px;
        }
        .section {
            width: 100%;
            height: 150px;
        }
    </style>
</head>
<body>
    <div class="section" style="background-color: white(50); color: red;">
        <h2>THANKS TO USE OUR GEN AI REOMENDATION SYSTEM </h2>
    </div>
    <div class="section" style="background-color: white(50); color: red;">
        <h2>PLEASE VISIT AGAIN</h2>
    </div>
    
</body>
            </div>
""", unsafe_allow_html=True)
