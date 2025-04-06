from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import pandas as pd

app = Flask(__name__)

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text2text-generation", model="google/flan-t5-small")

# SHL Product Catalog (Mock Data)
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

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.json
    job_input = content.get("job_description", "")

    if not job_input.strip():
        return jsonify({"error": "Job description is empty."}), 400

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

    return jsonify({
        "recommendations": output,
        "top_matches": [
            {
                "name": a['name'],
                "role": a['role'],
                "seniority": a['seniority'],
                "skills": a['skills']
            } for a in top_assessments
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)
