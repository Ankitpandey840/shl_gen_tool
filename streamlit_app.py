import asyncio
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# ========== FIXES FOR DEPLOYMENT ==========
# Fix event loop issues (critical for Streamlit)
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

# Memory optimization (reduces crashes on free hosting)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ========== MODEL LOADING (Optimized for CPU) ==========
@st.cache_resource  # Cache models to avoid reloading
def load_models():
    # Using smaller models to fit free tier memory limits
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    generator = pipeline(
        "text2text-generation", 
        model="google/flan-t5-small",  # Smaller than flan-t5-base
        device=-1  # Force CPU
    )
    return embedder, generator

embedder, generator = load_models()

# ========== SHL ASSESSMENT CATALOG ==========
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

# Precompute embeddings (cached for performance)
@st.cache_data
def get_embeddings():
    return embedder.encode(catalog['text'].tolist(), convert_to_tensor=True)

catalog_embeddings = get_embeddings()

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="SHL GenAI Assessment Tool", layout="wide")

# Custom CSS (your existing styling remains unchanged)
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

# Navbar (unchanged)
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

# Header
st.markdown("<h1 style='color:#4b2e83;'>SHL</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="main">
<h1> SHL GenAI Assessment Recommendation Tool</h1>
<p>Welcome to the <strong>SHL GenAI Assessment Recommendation Tool</strong>! </p>
<p>Paste a job description below, and let our AI recommend the best-fit assessments from our catalog based on the role, seniority, and skills required.</p>
""", unsafe_allow_html=True)

# Input Area
job_input = st.text_area("Paste the Job Description or Role Info:", height=200)

# Generate Button
if st.button("Generate Recommendations"):
    if not job_input.strip():
        st.warning("Please enter a job description to proceed.")
    else:
        with st.spinner("Analyzing with AI..."):
            try:
                # Compute similarity
                input_embedding = embedder.encode(job_input, convert_to_tensor=True)
                similarities = util.pytorch_cos_sim(input_embedding, catalog_embeddings)[0]
                top_k = min(3, len(similarities))
                top_results = similarities.topk(k=top_k)

                # Prepare retrieved info
                retrieved_info = ""
                top_assessments = []
                for score, idx in zip(top_results.values, top_results.indices):
                    item = catalog.iloc[idx.item()]
                    top_assessments.append(item)
                    retrieved_info += f"- {item['name']} ({item['role']}, {item['seniority']}): {item['skills']}\n"

                # Generate AI explanation
                prompt = f"""Job Description: {job_input}\n\nAvailable Assessments:\n{retrieved_info}\n\nRecommend the most suitable assessments from the list above for this job description, explaining your reasoning in 2-3 sentences."""
                output = generator(prompt, max_length=300, do_sample=False)[0]['generated_text']

                # Display results
                st.subheader("AI Recommendations:")
                st.success(output)

                st.markdown("---")
                st.markdown("<h4>Top Matching Assessments:</h4>", unsafe_allow_html=True)
                for i, a in enumerate(top_assessments, 1):
                    st.markdown(f"**{i}. {a['name']}**<br>• Role: {a['role']}<br>• Seniority: {a['seniority']}<br>• Skills: {a['skills']}", unsafe_allow_html=True)

                # Download report
                report_md = f"### SHL GenAI Recommendations\n\n**Job Description:**\n{job_input}\n\n**AI Recommendation:**\n{output}\n\n**Top Matches:**\n" + "\n".join([f"{i+1}. {a['name']} ({a['role']}, {a['seniority']}): {a['skills']}" for i, a in enumerate(top_assessments)])
                st.download_button(
                    label="Download Recommendations",
                    data=report_md,
                    file_name="shl_genai_recommendations.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f" Error: {str(e)}")
                st.info("If this persists, try a shorter job description or check back later.")

# Footer
st.markdown("""
<hr>
<p style="text-align:center;font-size:0.9em;">Made with using <a href="https://streamlit.io/" target="_blank">Streamlit</a> and <a href="https://huggingface.co/" target="_blank">Hugging Face</a>.</p>
""", unsafe_allow_html=True)
