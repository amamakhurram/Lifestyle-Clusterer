import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import base64
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(
    page_title="Spending Quest - Class Analyzer",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add glowing animated header style
st.markdown("""
<style>
/* ✨ Glowing Header Animation */
@keyframes glow {
  0% { text-shadow: 0 0 5px #fcd34d, 0 0 10px #facc15, 0 0 20px #facc15, 0 0 40px #facc15; }
  50% { text-shadow: 0 0 10px #facc15, 0 0 20px #facc15, 0 0 30px #facc15, 0 0 50px #facc15; }
  100% { text-shadow: 0 0 5px #fcd34d, 0 0 10px #facc15, 0 0 20px #facc15, 0 0 40px #facc15; }
}
.glow-header {
    font-size: 3em;
    font-weight: bold;
    color: #fcd34d;
    text-align: center;
    animation: glow 2s ease-in-out infinite;
    font-family: 'Courier New', monospace;
    margin-bottom: 20px;
}

/* 🌑 Background & Base Theme */
.main {
    background-color: #1e1e2f;
    color: #e0e0f0;
    font-family: 'Courier New', monospace;
}

/* 🟪 Buttons */
.stButton>button {
    background-color: #8b00ff;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    border: 2px solid #aa00ff;
    transition: transform 0.2s ease, background-color 0.3s ease;
    box-shadow: 0 0 10px rgba(138, 43, 226, 0.4);
}
.stButton>button:hover {
    background-color: #aa00ff;
    transform: scale(1.05);
}

/* 📘 Sidebar */
.sidebar .sidebar-content {
    background-color: #292945;
    color: white;
}

/* 🏷️ Headings (excluding glowing h1) */
.stMarkdown h2, .stMarkdown h3 {
    background: linear-gradient(90deg, #fcd34d, #facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}

/* 📜 Paragraph Text */
.stMarkdown p {
    color: #dcdcff;
    font-size: 1.1em;
    line-height: 1.6;
    margin-bottom: 1em;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

/* 🧾 List Items */
.stMarkdown ul li {
    color: #e8e8ff;
    font-size: 1.05em;
    margin-bottom: 0.6em;
    list-style-type: '🎯 ';
    padding-left: 0.5em;
    text-shadow: 0 1px 1px rgba(0,0,0,0.2);
}

/* 💬 Tooltip Styling */
.inline-tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
    font-size: 1em;
    color: #facc15;
    vertical-align: middle;
    transition: transform 0.2s;
}
.inline-tooltip:hover {
    transform: scale(1.15);
}
.inline-tooltip .inline-tooltip-text {
    visibility: hidden;
    opacity: 0;
    width: max-content;
    max-width: 200px;
    background: #000;
    color: #fff;
    text-align: left;
    font-size: 0.85em;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    top: 120%;
    z-index: 10;
    transition: opacity 0.2s;
    padding: 6px 10px;
    border-radius: 6px;
    white-space: pre-line;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    pointer-events: none;
}
.inline-tooltip:hover .inline-tooltip-text {
    visibility: visible;
    opacity: 1;
    pointer-events: auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glow-header">🧙‍♂️ Spending Quest - Class Analyzer</div>
""", unsafe_allow_html=True)

def tooltip(text):
    return f'''
    <span class="inline-tooltip">
        🎮
        <span class="inline-tooltip-text">{text}</span>
    </span>
    '''

def generate_mock_data():
    np.random.seed(42)
    data = {
        'Groceries': np.random.normal(500, 100, 30),
        'Dining': np.random.normal(300, 50, 30),
        'Shopping': np.random.normal(400, 150, 30),
        'Utilities': np.random.normal(200, 30, 30),
        'Entertainment': np.random.normal(250, 75, 30),
        'Savings': np.random.normal(1000, 200, 30)
    }
    return pd.DataFrame(data)

def validate_csv(df):
    required_columns = ['Groceries', 'Dining', 'Shopping', 'Utilities', 'Entertainment', 'Savings']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("🛑 Missing stats! Ensure your file includes all required categories.")
    if df.isnull().any().any():
        raise ValueError("⚠️ Missing values detected. Please heal your data!")
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"⚠️ '{col}' contains non-numeric values. Numbers only in this realm!")

def find_optimal_clusters(data, max_clusters=10):
    wcss = []
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    differences = np.diff(wcss)
    differences2 = np.diff(differences)
    elbow_point = np.argmax(differences2) + 2
    return elbow_point

def create_radar_chart(df, cluster_labels, user_index):
    categories = df.columns.drop('Cluster')
    user_values = df.iloc[user_index].drop('Cluster')
    cluster_avg = df[df['Cluster'] == cluster_labels[user_index]].mean().drop('Cluster')
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cluster_avg,
        theta=categories,
        fill='toself',
        name='Guild Average',
        line_color='rgba(139,0,255,0.5)',
        fillcolor='rgba(139,0,255,0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Your Stats',
        line_color='rgba(250,204,21,1)',
        fillcolor='rgba(250,204,21,0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(user_values), max(cluster_avg)) * 1.2]
            )
        ),
        showlegend=True,
        title="⚔️ Stat Comparison: You vs Guild",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_lifestyle_description(cluster_label, n_clusters):
    if n_clusters == 3:
        classes = {
            0: {'label': '🛡️ Frugal Knight', 'description': 'Steadfast in saving, focused on essentials. A budget tactician!'},
            1: {'label': '⚖️ Balanced Adventurer', 'description': 'A harmonious spender, you enjoy the quest while preserving potions (savings).'},
            2: {'label': '💎 Lavish Sorcerer', 'description': 'You wield wealth like magic—spending freely and enjoying the fantasy realm!'}
        }
        return classes.get(cluster_label, {'label': f'Class {cluster_label+1}', 'description': 'An unclassified hero of spending.'})
    else:
        return {'label': f'Class {cluster_label + 1}', 'description': 'A mysterious adventurer with unknown patterns.'}

# Section: Game Start
def section_home():
    st.markdown("""
    <h1>♂️ Welcome, Player!</h1>
    <p>Embark on a quest to uncover your spending class!</p>
    <ul>
    <li>📁 Load your file or use demo data</li>
    <li>⚙️ Prepare your stats with preprocessing</li>
    <li>🧠 Use Guild Sorting (K-Means) to classify your playstyle</li>
    <li>📊 Visualize and download your character sheet</li>
    </ul>
    """, unsafe_allow_html=True)

# Section: Data Upload
def section_data_upload(session_state):
    st.markdown(f"""<h3>📁 Load File {tooltip("Upload your spending CSV — categories: Groceries, Dining, Shopping, etc.")}</h3>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV", type=['csv'], key="file_uploader")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            validate_csv(df)
            st.success("📂 Save file loaded!")
            session_state['df'] = df
            session_state['uploaded'] = True
        except Exception as e:
            st.error(f"Error: {str(e)}")
            session_state['df'] = None
    else:
        st.info("⚔️ No save file? Try the demo mode below.")
        session_state['df'] = generate_mock_data()
        session_state['uploaded'] = False
    st.markdown(f"""<h4>📜 Your Stats {tooltip("This is your raw spending data.")}</h4>""", unsafe_allow_html=True)
    st.dataframe(session_state['df'])

# Section: Preprocessing
def section_preprocessing(session_state):
    st.markdown("<h3>⚙️ Preprocessing Chamber</h3>", unsafe_allow_html=True)
    if session_state.get('df') is not None:
        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(session_state['df'])
            scaled_df = pd.DataFrame(scaled, columns=session_state['df'].columns)
            session_state['scaled_df'] = scaled_df
            st.markdown(f"<h4>🔧 Normalized Stats {tooltip('Stats are now balanced using StandardScaler.')}</h4>", unsafe_allow_html=True)
            st.dataframe(scaled_df)
        except Exception as e:
            st.error(f"⚠️ Alchemy failed: {str(e)}")
    else:
        st.info("📁 You must first load your save file.")

# Section: Clustering
def section_clustering(session_state):
    st.markdown("<h2>🧠 Guild Sorting Ceremony</h2>", unsafe_allow_html=True)
    if session_state.get('scaled_df') is not None:
        try:
            data = session_state['scaled_df'].values
            n_clusters = find_optimal_clusters(data)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data)
            session_state['df']['Cluster'] = labels
            session_state['scaled_df']['Cluster'] = labels
            lifestyle_info = get_lifestyle_description(labels[-1], n_clusters)
            session_state.update({
                'cluster_labels': labels,
                'n_clusters': n_clusters,
                'lifestyle_info': lifestyle_info
            })
            st.markdown(f"""
            <div style='background-color:#292945; padding:20px; border-radius:15px; margin-bottom:20px;'>
                <span style='font-size:1.2em; color:#facc15;'>
                🎉 You have been sorted into the <b>{lifestyle_info['label']}</b>!
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<b>📝 Description:</b> {lifestyle_info['description']}", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Sorting spell failed: {str(e)}")
    else:
        st.info("⚠️ You need to enter the Preprocessing Chamber first.")

# Section: Visualization
def section_visualization(session_state):
    st.markdown("<h2>📊 Character Sheet</h2>", unsafe_allow_html=True)
    if session_state.get('df') is not None and session_state.get('cluster_labels') is not None:
        col1, col2 = st.columns(2)
        with col1:
            radar_fig = create_radar_chart(session_state['df'], session_state['cluster_labels'], -1)
            st.plotly_chart(radar_fig, use_container_width=True)
        with col2:
            bar_fig = px.bar(
                session_state['df'].drop('Cluster', axis=1).iloc[-1],
                title="💰 Gold Distribution",
                labels={'value': 'Amount ($)', 'index': 'Category'},
                color_discrete_sequence=['#facc15']
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        if st.button("💾 Export Character Sheet"):
            export_df = pd.DataFrame()
            for col in session_state['df'].columns.drop('Cluster'):
                export_df[f'Original_{col}'] = session_state['df'][col]
            for col in session_state['scaled_df'].columns.drop('Cluster'):
                export_df[f'Normalized_{col}'] = session_state['scaled_df'][col]
            export_df['Cluster_Label'] = session_state['df']['Cluster']
            export_df['Lifestyle_Class'] = [get_lifestyle_description(lbl, session_state['n_clusters'])['label'] for lbl in session_state['df']['Cluster']]
            buffer = io.StringIO()
            export_df.to_csv(buffer, index=False)
            st.download_button("🧾 Download CSV", buffer.getvalue(), file_name="spending_class_report.csv", mime="text/csv")
    else:
        st.info("📍 Finish the quest steps to unlock your stats!")

# 🧙‍♂️ Main Game Loop
def main():
    st.sidebar.title("🗺️ Quest Menu")
    section = st.sidebar.selectbox("Choose Your Path:", ("Home", "Load Save File", "Preprocessing Chamber", "Guild Sorting", "Character Sheet"))
    for key in ['df', 'scaled_df', 'cluster_labels', 'n_clusters', 'lifestyle_info', 'uploaded']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'uploaded' else False
    session_state = st.session_state
    if section == "Home":
        section_home()
    elif section == "Load Save File":
        section_data_upload(session_state)
    elif section == "Preprocessing Chamber":
        section_preprocessing(session_state)
    elif section == "Guild Sorting":
        section_clustering(session_state)
    elif section == "Character Sheet":
        section_visualization(session_state)

if __name__ == "__main__":
    main()
