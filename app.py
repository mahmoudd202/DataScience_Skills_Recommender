import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re
from gensim.models import Word2Vec
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import math
import logging
# Set page configuration
plt.style.use('ggplot')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("job_skills_recommender")

# Load datasets with error handling
try:
    logger.info("Loading datasets 🚀")
    job_skills_df = pd.read_csv("job_skills.csv")
    job_postings_df = pd.read_csv("linkedin_job_postings.csv")

    merged_df = job_postings_df.merge(job_skills_df, on="job_link", how="left")
    merged_df = merged_df[['job_title', 'job_skills']]
    merged_df.dropna(subset=['job_skills'], inplace=True)
    logger.info(f"Dataset loaded with {len(merged_df)} job postings")
except Exception as e:
    logger.error(f"Error loading datasets: {e}")
    raise

job_title_skills = {}
for _, row in merged_df.iterrows():
    job_title = row['job_title'].lower()
    skills = [skill.strip() for skill in row['job_skills'].split(',')]
    job_title_skills.setdefault(job_title, []).extend(skills)

job_titles = list(job_title_skills.keys())
all_skills = [', '.join(skills) for skills in job_title_skills.values()]

unique_jobs_df = pd.DataFrame({
    'job_title': job_titles,
    'all_skills': all_skills
})
print(f"Unique job titles: {len(unique_jobs_df)}")

tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                   token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
                                   stop_words='english',
                                   ngram_range=(1, 2))

job_title_vectors = tfidf_vectorizer.fit_transform(unique_jobs_df['job_title'])
print(f"Job title vector shape: {job_title_vectors.shape}")

# Prepare data for co-occurrence map
all_skills_lists = []
for skills_text in unique_jobs_df['all_skills']:
    skills_list = [re.sub(r'[^\w\s]', '', skill).lower().strip() for skill in skills_text.split(',')]
    skills_list = [skill for skill in skills_list if skill]  # Remove empty strings
    all_skills_lists.append(skills_list)

# Load pre-trained Word2Vec model instead of training it
try:
    logger.info("Loading pre-trained Word2Vec model 🔄")
    word2vec_model = Word2Vec.load("word2vec_model.model")
    logger.info("Word2Vec model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Word2Vec model: {e}")
    raise

# Build co-occurrence map
co_occurrence_map = defaultdict(Counter)
for skill_list in all_skills_lists:
    cleaned = [re.sub(r'[^\w\s]', '', s).lower().strip() for s in skill_list if s]
    for skill in set(cleaned):
        for other in set(cleaned):
            if skill != other:
                co_occurrence_map[skill][other] += 1


class JobSkillsRecommender:
    def __init__(self, job_title_vectors, job_titles, job_skills, vectorizer, word2vec_model):
        self.job_title_vectors = job_title_vectors
        self.job_titles = job_titles
        self.job_skills = job_skills
        self.vectorizer = vectorizer
        self.word2vec_model = word2vec_model
        self.all_skills_set = set()
        for skills in job_skills:
            self.all_skills_set.update(self._clean(skill) for skill in skills.split(', '))

    def _clean(self, text):
        return re.sub(r'[^\w\s]', '', text).lower().strip()

    def get_recommended_skills(self, job_title, top_n=10, similarity_threshold=0.3):
        job_title = job_title.lower()
        similar_jobs = []

        if job_title in self.job_titles:
            i = self.job_titles.index(job_title)
            skill_counts = Counter(self.job_skills[i].split(', '))
            skills = [skill for skill, _ in skill_counts.most_common(top_n)]
            return skills, skill_counts, [job_title], self.get_top_related_skills(skills)

        job_title_vector = self.vectorizer.transform([job_title])
        similarities = cosine_similarity(job_title_vector, self.job_title_vectors).flatten()
        similar_indices = np.where(similarities >= similarity_threshold)[0][:5]

        if len(similar_indices) == 0:
            all_skills_flat = ', '.join(self.job_skills).split(', ')
            skill_counts = Counter(all_skills_flat)
            skills = [skill for skill, _ in skill_counts.most_common(top_n)]
            return skills, skill_counts, [], self.get_top_related_skills(skills)

        for i in similar_indices:
            similar_jobs.append(self.job_titles[i])

        all_skills = []
        for i in similar_indices:
            all_skills.extend(self.job_skills[i].split(', '))

        skill_counts = Counter(all_skills)
        skills = [skill for skill, _ in skill_counts.most_common(top_n)]
        return skills, skill_counts, similar_jobs, self.get_top_related_skills(skills)

    def get_top_related_skills(self, seed_skills, topn_related=10):
        seen = set(self._clean(s) for s in seed_skills)
        related_scores = defaultdict(float)

        for seed in seed_skills:
            base = self._clean(seed)
            if base not in self.word2vec_model.wv or base not in self.co_occurrence_map:
                continue

            # Word2Vec top similar
            similar = self.word2vec_model.wv.most_similar(base, topn=20)
            for related, sim in similar:
                clean_related = self._clean(related)
                if clean_related in seen:
                    continue

                # Keep only skills that co-occur with the seed skill
                if clean_related in self.co_occurrence_map[base]:
                    # Score = similarity * co-occurrence count
                    related_scores[clean_related] += sim * math.log1p(self.co_occurrence_map[base][clean_related]) #Reduces bias toward frequent skills

        # Sort and return top N
        return sorted(related_scores.items(), key=lambda x: -x[1])[:topn_related]


# Initialize the recommender
recommender = JobSkillsRecommender(
    job_title_vectors,
    unique_jobs_df['job_title'].tolist(),
    unique_jobs_df['all_skills'].tolist(),
    tfidf_vectorizer,
    word2vec_model
)

recommender.co_occurrence_map = co_occurrence_map


# Create visualizations for Gradio UI
# Add error handling for better stability
def create_skill_frequency_chart(skill_counts, title="Top Skills by Frequency"):
    if not skill_counts:
        fig = px.bar(
            pd.DataFrame({'Skill': ['No Data'], 'Frequency': [0]}),
            x='Frequency',
            y='Skill',
            title="No skill data available"
        )
        return fig

    skills_df = pd.DataFrame({
        'Skill': list(skill_counts.keys()),
        'Frequency': list(skill_counts.values())
    }).sort_values('Frequency', ascending=False).head(15)

    fig = px.bar(
        skills_df,
        x='Frequency',
        y='Skill',
        title=title,
        orientation='h',
        color='Frequency',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Frequency in Job Postings",
        yaxis_title="Skills",
        font=dict(size=12)
    )

    return fig


def create_skills_network(seed_skills, related_skills, title="Skills Relationship Network"):
    G = nx.Graph()

    # Add nodes
    for skill in seed_skills:
        G.add_node(skill, type="seed", size=20)

    # Add related skills as nodes
    for skill, score in related_skills:
        if skill not in G:
            G.add_node(skill, type="related", size=10 + score * 5)

    # Add edges
    for skill, score in related_skills:
        for seed in seed_skills:
            seed_clean = recommender._clean(seed)
            skill_clean = recommender._clean(skill)

            if skill_clean in recommender.co_occurrence_map.get(seed_clean, {}):
                weight = recommender.co_occurrence_map[seed_clean][skill_clean]
                G.add_edge(seed, skill, weight=weight)

    # Generate layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)

    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge].get('weight', 1)

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=min(weight / 5, 3), color='rgba(150,150,150,0.3)'),
            hoverinfo='none',
            mode='lines')

        edge_traces.append(edge_trace)

    # Create node traces
    node_trace_seed = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            color='rgba(41, 128, 185, 0.8)',
            size=15,
            line=dict(width=1, color='rgb(41, 128, 185)')
        )
    )

    node_trace_related = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            color='rgba(192, 57, 43, 0.7)',
            size=10,
            line=dict(width=1, color='rgb(192, 57, 43)')
        )
    )

    # Add node data
    for node in G.nodes():
        x, y = pos[node]
        node_type = G.nodes[node].get('type', 'related')

        if node_type == 'seed':
            node_trace_seed.x = node_trace_seed.x + (x,)
            node_trace_seed.y = node_trace_seed.y + (y,)
            node_trace_seed.text = node_trace_seed.text + (node,)
        else:
            node_trace_related.x = node_trace_related.x + (x,)
            node_trace_related.y = node_trace_related.y + (y,)
            node_trace_related.text = node_trace_related.text + (node,)

    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace_seed, node_trace_related],
        layout=go.Layout(
            title=title,
            showlegend=False,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='rgb(248,248,248)'
        )
    )

    return fig

def get_job_recommendations(job_title, number_of_skills=10, similarity_threshold=0.3):
    recommended_skills, skill_counts, similar_jobs, related_skills = recommender.get_recommended_skills(
        job_title,
        top_n=number_of_skills,
        similarity_threshold=similarity_threshold
    )

    # Create output components
    job_html = ""
    if similar_jobs:
        job_html = "<div style='margin-bottom: 20px;'>"
        job_html += f"<h3>Similar Job Titles</h3>"
        job_html += "<ul style='list-style-type: disc; padding-left: 20px;'>"
        for job in similar_jobs:
            job_html += f"<li>{job}</li>"
        job_html += "</ul></div>"
    else:
        job_html = "<div style='margin-bottom: 20px;'>"
        job_html += f"<h3>No exact matches found</h3>"
        job_html += "<p>Showing general skill recommendations based on all jobs.</p>"
        job_html += "</div>"

    # Create skills table
    skills_html = "<div style='margin-bottom: 20px;'>"
    skills_html += f"<h3>Top {len(recommended_skills)} Recommended Skills</h3>"
    skills_html += "<table style='width: 100%; border-collapse: collapse;'>"
    skills_html += "<tr style='background-color: #f2f2f2;'><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Skill</th><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Frequency</th></tr>"

    for skill in recommended_skills:
        skills_html += f"<tr><td style='padding: 10px; border: 1px solid #ddd;'>{skill}</td><td style='padding: 10px; border: 1px solid #ddd;'>{skill_counts[skill]}</td></tr>"

    skills_html += "</table></div>"

    # Create related skills table
    related_html = ""
    if related_skills:
        related_html = "<div style='margin-bottom: 20px;'>"
        related_html += f"<h3>Top {len(related_skills)} Related Skills (by Word2Vec)</h3>"
        related_html += "<table style='width: 100%; border-collapse: collapse;'>"
        related_html += "<tr style='background-color: #f2f2f2;'><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Skill</th><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Relevance Score</th></tr>"

        for skill, score in related_skills:
            related_html += f"<tr><td style='padding: 10px; border: 1px solid #ddd;'>{skill}</td><td style='padding: 10px; border: 1px solid #ddd;'>{score:.3f}</td></tr>"

        related_html += "</table></div>"

    # Create visualizations
    freq_chart = create_skill_frequency_chart(skill_counts, title=f"Top Skills for '{job_title}'")
    network_chart = create_skills_network(recommended_skills, related_skills)

    html_output = html_output = f"""
            <div style='font-family: Arial, sans-serif; padding: 20px;'>

                <div style='margin-bottom: 25px;'>
                    <h2 style='color: #2c3e50;'>🔍 Similar Job Titles</h2>
                    {"<ul>" + "".join(f"<li>{job}</li>" for job in similar_jobs) + "</ul>" if similar_jobs else "<p style='color: #888;'>No exact matches. Showing general suggestions.</p>"}
                </div>

                <div style='margin-bottom: 25px;'>
                    <h2 style='color: #2c3e50;'>⭐ Top {len(recommended_skills)} Recommended Skills</h2>
                    <div style='display: flex; flex-wrap: wrap; gap: 10px;'>
                        {"".join(f"<div style='background-color:#ecf0f1;padding:10px 15px;border-radius:8px;border:1px solid #ccc;font-weight:bold;color:#2c3e50;'>{skill} <span style='color:#888;font-weight:normal'>(x{skill_counts[skill]})</span></div>" for skill in recommended_skills)}
                    </div>
                </div>

                <div style='margin-bottom: 25px;'>
                    <h2 style='color: #2c3e50;'>🧠 Related Skills (by Word2Vec)</h2>
                    <div style='display: flex; flex-wrap: wrap; gap: 10px;'>
                        {"".join(f"<div style='background-color:#f9f9f9;padding:10px 15px;border-radius:8px;border:1px solid #ddd;color:#34495e;'>{skill} <span style='font-size: 0.9em; color:#888;'>(score {score:.2f})</span></div>" for skill, score in related_skills)}
                    </div>
                </div>

            </div>
            """


    return html_output, freq_chart, network_chart


# Define custom CSS - stored in separate file for better IDE handling
def get_custom_css():
    return """
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f7f9fc;
    }

    .gradio-container {
        max-width: 100% !important;
    }

    .gr-form {
        flex-direction: column;
        gap: 12px;
    }

    h1 {
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }

    h2 {
        color: #34495e;
        font-weight: 500;
        margin-top: 1rem;
    }

    .gr-box {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .gr-button {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: background-color 0.3s ease !important;
    }

    .gr-button:hover {
        background-color: #2980b9 !important;
    }

    .gr-input {
        border-radius: 5px !important;
        border: 1px solid #ddd !important;
        padding: 10px !important;
    }

    .gr-input:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2) !important;
    }

    .gr-panel {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        background-color: white;
    }

    .gr-markdown p {
        margin: 0.5rem 0;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }

    th, td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }

    th {
        background-color: #f2f2f2;
        font-weight: 600;
    }

    tr:nth-child(even) {
        background-color: #f9f9f9;
    }

    .tabs {
        min-height: 25vh;
        padding-bottom: 40px;

    }

    .footer {
        position: relative;
        margin-top: 40px;
        text-align: center;
        padding: 10px 0;
        font-size: 0.85rem;
        color: #7f8c8d;
    }
    """


custom_css = get_custom_css()


# Add a theme selector
def create_interface():

    with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🚀 Job Skills Recommender
                ### Find the most relevant skills for your target job role
                """
            )

        gr.Markdown(
            """
            This tool analyzes a large dataset of job postings to recommend the most valuable skills for your career.
            Enter a job title below and discover which skills will help you stand out to employers!
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                job_title_input = gr.Textbox(
                    label="Enter Job Title",
                    placeholder="e.g., Data Scientist, Software Engineer, Product Manager...",
                    lines=1
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        skills_count = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=10,
                            step=1,
                            label="Number of Skills to Recommend"
                        )

                    with gr.Column(scale=1):
                        similarity_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Job Title Similarity Threshold"
                        )

                submit_btn = gr.Button("Get Skill Recommendations", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown(
                    """
                    ### How to use this tool:

                    1. Enter a job title you're interested in
                    2. Adjust the number of skills you want to see
                    3. Click "Get Skill Recommendations" to analyze
                    4. Explore the recommended skills and visualizations

                    This tool uses AI-powered analysis of job postings to suggest skills that are most relevant to your career goals.
                    """
                )

        with gr.Tabs() as tabs:
            with gr.TabItem("Overview"):
                html_output = gr.HTML(label="Recommended Skills")

            with gr.TabItem("Skill Frequency Chart"):
                freq_chart = gr.Plot(label="Skill Frequency")

            with gr.TabItem("Skills Network"):
                network_chart = gr.Plot(label="Skills Network Graph")

            with gr.TabItem("About"):
                gr.Markdown(
                    """
                    ### 📊 About the Data

                    This recommender system analyzes thousands of real job postings from LinkedIn to find the most frequently 
                    requested skills for each job title. It uses natural language processing (NLP) and machine learning 
                    to understand how skills relate to each other.

                    **Technical Details:**

                    - **Word2Vec Model**: Used to understand semantic relationships between skills
                    - **TF-IDF Vectorization**: For job title similarity matching
                    - **Network Analysis**: To visualize skill relationships and co-occurrences
                    - **Data Source**: LinkedIn job postings dataset

                    For technical roles, both hard skills (programming languages, tools) and soft skills 
                    (communication, teamwork) are considered.
                    """
                )

        gr.Markdown(
            """
            <div class="footer">
            Developed with tears using Gradio | Dataset sourced from LinkedIn job postings | Last updated: May 2025
            </div>
            """
        )

        # Set up the function call
        submit_btn.click(
            get_job_recommendations,
            inputs=[job_title_input, skills_count, similarity_slider],
            outputs=[html_output, freq_chart, network_chart]
        )

    return demo

demo = create_interface()