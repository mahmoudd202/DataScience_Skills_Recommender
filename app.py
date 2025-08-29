import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re
import os
import pickle
from gensim.models import Word2Vec
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import math
import logging
from sklearn.cluster import KMeans

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

model_path = "word2vec_model.model"

if os.path.exists(model_path):
    word2vec_model = Word2Vec.load(model_path)
    logger.info("Word2Vec model loaded successfully")
else:
    logger.info("Training new Word2Vec model...")
    word2vec_model = Word2Vec(
        sentences=all_skills_lists,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,
        workers=4
    )
    word2vec_model.save(model_path)
    logger.info("Word2Vec model trained and saved")

# Load or create co-occurrence map
if os.path.exists("co_occurrence_map.pkl"):
    with open("co_occurrence_map.pkl", "rb") as f:
        co_occurrence_map = pickle.load(f)
    logger.info("Co-occurrence map loaded")
else:
    # Create co-occurrence map
    logger.info("Creating co-occurrence map...")
    co_occurrence_map = defaultdict(Counter)
    for skill_list in all_skills_lists:
        cleaned = [re.sub(r'[^\w\s]', '', s).lower().strip() for s in skill_list if s]
        for skill in set(cleaned):
            for other in set(cleaned):
                if skill != other:
                    co_occurrence_map[skill][other] += 1

    # Save it
    with open("co_occurrence_map.pkl", "wb") as f:
        pickle.dump(co_occurrence_map, f)
    logger.info("Co-occurrence map created and saved")


class JobSkillsRecommender:
    def __init__(self, job_title_vectors, job_titles, job_skills, vectorizer, word2vec_model, co_occurrence_map):
        self.job_title_vectors = job_title_vectors
        self.job_titles = job_titles
        self.job_skills = job_skills
        self.vectorizer = vectorizer
        self.word2vec_model = word2vec_model
        self.co_occurrence_map = co_occurrence_map
        self.all_skills_set = set()
        for skills in job_skills:
            self.all_skills_set.update(self._clean(skill) for skill in skills.split(', '))

    def _clean(self, text):
        return re.sub(r'[^\w\s]', '', text).lower().strip()

    def is_job_title_relevant(self, job_title, similarity_threshold=0.1):
        """Check if the job title is relevant to our dataset"""
        job_title = job_title.lower().strip()

        # Check for exact match first
        if job_title in self.job_titles:
            return True, 1.0

        # Check similarity with existing job titles
        job_title_vector = self.vectorizer.transform([job_title])
        similarities = cosine_similarity(job_title_vector, self.job_title_vectors).flatten()
        max_similarity = np.max(similarities)

        return max_similarity >= similarity_threshold, max_similarity

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
            try:
                similar = self.word2vec_model.wv.most_similar(base, topn=20)
                for related, sim in similar:
                    clean_related = self._clean(related)
                    if clean_related in seen:
                        continue

                    # Keep only skills that co-occur with the seed skill
                    if clean_related in self.co_occurrence_map[base]:
                        # Score = similarity * co-occurrence count
                        related_scores[clean_related] += sim * math.log1p(self.co_occurrence_map[base][clean_related])
            except KeyError:
                continue

        # Sort and return top N
        return sorted(related_scores.items(), key=lambda x: -x[1])[:topn_related]


# === PART 1 === Huggingface ds_salaries.csv + TF-IDF + KMeans clustering ===
try:
    logger.info("Loading Huggingface ds_salaries.csv 🚀")
    huggingface_df = pd.read_csv("ds_salaries.csv")
    huggingface_df = huggingface_df[['job_title', 'experience_level', 'salary_in_usd']]
    logger.info(f"Huggingface dataset loaded with {len(huggingface_df)} entries")

    hug_tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                           token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
                                           stop_words='english',
                                           ngram_range=(1, 2))
    hug_job_title_vectors = hug_tfidf_vectorizer.fit_transform(huggingface_df['job_title'])

    hug_kmeans = KMeans(n_clusters=5, random_state=42)
    huggingface_df['cluster'] = hug_kmeans.fit_predict(hug_job_title_vectors)

    mean_salary_per_cluster_level = huggingface_df.groupby(['cluster', 'experience_level'])['salary_in_usd'].mean()

    logger.info("Huggingface cluster + level salary table ready ✅")

except Exception as e:
    logger.error(f"Error loading Huggingface salary data: {e}")
    # Create dummy data to prevent crashes
    huggingface_df = pd.DataFrame({
        'job_title': ['Data Scientist', 'Software Engineer'],
        'experience_level': ['MI', 'SE'],
        'salary_in_usd': [100000, 120000]
    })
    hug_tfidf_vectorizer = TfidfVectorizer()
    hug_job_title_vectors = hug_tfidf_vectorizer.fit_transform(huggingface_df['job_title'])
    hug_kmeans = KMeans(n_clusters=2, random_state=42)
    huggingface_df['cluster'] = hug_kmeans.fit_predict(hug_job_title_vectors)
    mean_salary_per_cluster_level = huggingface_df.groupby(['cluster', 'experience_level'])['salary_in_usd'].mean()

# Initialize the recommender
recommender = JobSkillsRecommender(
    job_title_vectors,
    unique_jobs_df['job_title'].tolist(),
    unique_jobs_df['all_skills'].tolist(),
    tfidf_vectorizer,
    word2vec_model,
    co_occurrence_map
)


# Create visualizations for Gradio UI
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

    if len(G.nodes()) == 0:
        # Return empty plot if no nodes
        fig = go.Figure()
        fig.update_layout(title="No network data available")
        return fig

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


def estimate_salary(job_title_input, experience_level_input):
    try:
        job_title_input = job_title_input.strip().lower()

        # First check if the job title is relevant using our main recommender
        is_relevant, similarity_score = recommender.is_job_title_relevant(job_title_input, similarity_threshold=0.1)

        if not is_relevant:
            return "💰 Estimated Salary: job title not found in our database. Please try a more common job title. "

        job_vector = hug_tfidf_vectorizer.transform([job_title_input])
        predicted_cluster = hug_kmeans.predict(job_vector)[0]

        try:
            estimated_salary = mean_salary_per_cluster_level.loc[(predicted_cluster, experience_level_input)]
            estimated_salary = round(estimated_salary, 2)
            logger.info(
                f"Estimated salary for '{job_title_input}' at level '{experience_level_input}' → ${estimated_salary}")
            return f"💰 Estimated Salary: ${estimated_salary:,.2f} (Confidence: {similarity_score:.2f})"
        except KeyError:
            logger.warning(f"No salary data for cluster {predicted_cluster} and level '{experience_level_input}'")
            return f"💰 Estimated Salary: Not available for this experience level"
    except Exception as e:
        logger.error(f"Error estimating salary: {e}")
        return f"💰 Estimated Salary: Error occurred during estimation"


def get_comprehensive_analysis(job_title, experience_level, number_of_skills=10, similarity_threshold=0.3):
    """Combined function that returns both skills recommendations and salary estimation"""
    try:
        # Get skills recommendations
        recommended_skills, skill_counts, similar_jobs, related_skills = recommender.get_recommended_skills(
            job_title,
            top_n=number_of_skills,
            similarity_threshold=similarity_threshold
        )

        # Show salary only if there are similar jobs
        if similar_jobs:
            salary_estimate = estimate_salary(job_title, experience_level)
        else:
            salary_estimate = "💰 Estimated Salary: Not shown due to lack of similar job matches"


        # Create visualizations
        freq_chart = create_skill_frequency_chart(skill_counts, title=f"Top Skills for '{job_title}'")
        network_chart = create_skills_network(recommended_skills, related_skills)

        html_output = f"""
        <div style='font-family: Arial, sans-serif; padding: 20px;'>

            <div style='margin-bottom: 25px; padding: 15px; background-color: #e8f5e8; border-radius: 10px; border-left: 5px solid #27ae60;'>
                <h2 style='color: #27ae60; margin-top: 0;'>{salary_estimate}</h2>
            </div>

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
    except Exception as e:
        error_msg = f"<div style='color: red; padding: 20px;'>Error occurred: {str(e)}</div>"
        empty_fig = go.Figure()
        return error_msg, empty_fig, empty_fig


# Define custom CSS
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


# Create interface
def create_interface():
    with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🚀 Job Skills Recommender & Salary Estimator
                ### Find the most relevant skills and estimated salary for your target job role
                """
            )

        gr.Markdown(
            """
            This tool analyzes a large dataset of job postings to recommend the most valuable skills for your career and estimates potential salary.
            Enter a job title below and discover which skills will help you stand out to employers, plus get salary insights!
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
                        experience_level_input = gr.Dropdown(
                            label="Experience Level",
                            choices=[
                                ('Entry Level', 'EN'),
                                ('Mid Level', 'MI'),
                                ('Senior Level', 'SE'),
                                ('Executive Level', 'EX')
                            ],
                            value='EN',
                            interactive=True,
                            info="Please select your experience level"
                        )

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

                submit_btn = gr.Button("Get Complete Analysis", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown(
                    """
                    ### How to use this tool:

                    1. **Enter a job title** you're interested in
                    2. **Select your experience level** for salary estimation
                    3. **Adjust settings** (optional):
                       - Number of skills to recommend
                       - Similarity threshold for job matching
                    4. **Click "Get Complete Analysis"** to see:
                       - Estimated salary range
                       - Recommended skills
                       - Interactive visualizations

                    This tool uses AI-powered analysis of real job postings to provide insights that are most relevant to your career goals.
                    """
                )

        with gr.Tabs() as tabs:
            with gr.TabItem("📊 Complete Analysis"):
                html_output = gr.HTML(label="Analysis Results")

            with gr.TabItem("📈 Skill Frequency Chart"):
                freq_chart = gr.Plot(label="Skill Frequency")

            with gr.TabItem("🕸️ Skills Network"):
                network_chart = gr.Plot(label="Skills Network Graph")

            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
                    ### 📊 About the Data & Methods

                    This recommender system analyzes thousands of real job postings from LinkedIn and salary data to provide comprehensive career insights.

                    **Key Features:**

                    - **Skills Recommendation**: Uses TF-IDF vectorization and Word2Vec to find the most relevant skills
                    - **Salary Estimation**: K-means clustering on job titles with experience level adjustment
                    - **Relevance Filtering**: Only provides salary estimates for job titles found in our database
                    - **Network Analysis**: Visualizes how skills relate to each other based on co-occurrence

                    **Technical Details:**

                    - **Word2Vec Model**: Semantic relationships between skills
                    - **TF-IDF Vectorization**: Job title similarity matching
                    - **K-means Clustering**: Groups similar roles for salary estimation
                    - **Co-occurrence Analysis**: Skills that frequently appear together
                    - **Data Sources**: LinkedIn job postings + salary datasets

                    **Experience Levels:**
                    - **EN (Entry Level)**: 0-2 years of experience
                    - **MI (Mid Level)**: 2-5 years of experience  
                    - **SE (Senior Level)**: 5+ years of experience
                    - **EX (Executive Level)**: Leadership/C-level positions

                    For best results, use common job titles like "Data Scientist", "Software Engineer", "Product Manager", etc.
                    """
                )

        gr.Markdown(
            """
            <div class="footer">
            Developed with ❤️ using Gradio | Dataset sourced from LinkedIn job postings & salary data | Last updated: May 2025
            </div>
            """
        )

        # Set up the function calls - now using the combined function
        submit_btn.click(
            get_comprehensive_analysis,
            inputs=[job_title_input, experience_level_input, skills_count, similarity_slider],
            outputs=[html_output, freq_chart, network_chart]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
