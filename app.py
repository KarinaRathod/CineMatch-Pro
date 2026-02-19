import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineMatch Pro",
    page_icon="🎬",
    layout="wide"
)

# ---------------- ADVANCED CUSTOM CSS ----------------
st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #0F1116;
}

/* Sidebar Background */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
    padding-top: 10px;
}

/* ===== CINEMATIC SIDEBAR HEADER ===== */
[data-testid="stSidebar"] h1 {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #E50914, #FF5F6D, #FFFFFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
    margin-bottom: 0px;
    animation: glow 3s ease-in-out infinite alternate;
}

/* Decorative underline */
[data-testid="stSidebar"] h1::after {
    content: "";
    display: block;
    width: 60px;
    height: 4px;
    margin-top: 8px;
    border-radius: 5px;
    background: linear-gradient(90deg, #E50914, #FF5F6D);
}

/* Glow Animation */
@keyframes glow {
    from {
        filter: drop-shadow(0 0 4px rgba(229, 9, 20, 0.4));
    }
    to {
        filter: drop-shadow(0 0 10px rgba(255, 95, 109, 0.7));
    }
}

/* Sidebar Subtitle */
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.95rem;
    color: #C9D1D9;
    font-weight: 400;
    margin-top: -5px;
    margin-bottom: 15px;
    letter-spacing: 0.3px;
}

/* Primary Button */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    width: 100%;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
}

/* Movie Card */
.movie-card {
    background: linear-gradient(145deg, #1E232B, #16181D);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px;
    height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.2s, border-color 0.2s;
}

.movie-card:hover {
    transform: translateY(-5px);
    border-color: #E50914;
    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
}

.card-title {
    color: #FFFFFF;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 5px;
}

.card-meta {
    color: #8B949E;
    font-size: 0.85rem;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.match-badge {
    background-color: rgba(35, 134, 54, 0.2);
    color: #3FB950;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: bold;
    border: 1px solid rgba(35, 134, 54, 0.3);
}

.plot-preview {
    color: #C9D1D9;
    font-size: 0.8rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    font-style: italic;
}

/* Main Header */
.main-header {
    text-align: center;
    font-weight: 700;
    background: linear-gradient(90deg, #FFFFFF, #E50914);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATA ENGINE ----------------
@st.cache_data
def get_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")

    movies = movies[['movie_id', 'title', 'overview',
                     'genres', 'keywords', 'cast',
                     'crew', 'vote_average', 'vote_count']]

    movies.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(
        lambda x: [i['name'] for i in ast.literal_eval(x)][:3]
    )

    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return i['name']
        return "Unknown"

    movies['director'] = movies['crew'].apply(fetch_director)

    movies['tags'] = movies['overview'].apply(lambda x: x.split()) + \
                     movies['genres'] + \
                     movies['keywords'] + \
                     movies['cast'] + \
                     movies['director'].apply(lambda x: [x])

    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

    return movies


df = get_data()

@st.cache_resource
def compute_sim(data):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['tags']).toarray()
    return cosine_similarity(vectors)

similarity = compute_sim(df)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("CineMatch Pro")
    st.markdown(
        "<p>✨ Your AI-powered cinematic companion</p>",
        unsafe_allow_html=True
    )
    st.divider()

    selected_movie = st.selectbox(
        "🔍 Search for a movie",
        df['title'].values
    )

    if st.button("Find Matches"):
        st.session_state['search_trigger'] = True
        st.session_state['selected_movie'] = selected_movie

    st.markdown("---")
    st.caption(
        "Algorithm: Cosine Similarity on Overview + Genres + Cast + Director."
    )

# ---------------- MAIN AREA ----------------
if 'search_trigger' in st.session_state and st.session_state['search_trigger']:

    movie = st.session_state['selected_movie']

    try:
        movie_index = df[df['title'] == movie].index[0]
        distances = sorted(
            list(enumerate(similarity[movie_index])),
            reverse=True,
            key=lambda x: x[1]
        )

        current_movie = df.iloc[movie_index]

        st.markdown(f"### 🎯 Because you liked **{movie}**")

        with st.expander("Show Source Movie Details"):
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                st.metric("Rating", f"⭐ {current_movie['vote_average']}/10")
            with c2:
                st.metric("Director", current_movie['director'])
            with c3:
                st.write(f"**Plot:** {current_movie['overview']}")

        st.markdown("---")

        cols = st.columns(4)

        for i, col in enumerate(cols):
            rec_idx = distances[i+1][0]
            rec_row = df.iloc[rec_idx]
            match_score = int(distances[i+1][1] * 100)

            with col:
                st.markdown(f"""
                <div class="movie-card">
                    <div>
                        <div class="card-title">{rec_row['title']}</div>
                        <div class="card-meta">
                            <span>⭐ {rec_row['vote_average']}</span>
                            <span>•</span>
                            <span class="match-badge">{match_score}% Match</span>
                        </div>
                        <div class="plot-preview">
                            "{rec_row['overview']}"
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("More Info", key=f"btn_{i}"):
                    st.toast(
                        f"Director: {rec_row['director']} | Genres: {', '.join(rec_row['genres'])}"
                    )

    except IndexError:
        st.error("Movie not found in database.")

else:
    st.markdown(
        "<h1 class='main-header'>Discover Cinematic Gems</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #8B949E;'>Select a movie from the sidebar to generate personalized recommendations.</p>",
        unsafe_allow_html=True
    )

    st.markdown("### 🔥 Top Rated in Database")

    top_movies = df[df['vote_count'] > 1000] \
        .sort_values('vote_average', ascending=False) \
        .head(4)

    d_cols = st.columns(4)

    for i, (_, row) in enumerate(top_movies.iterrows()):
        with d_cols[i]:
            st.markdown(f"""
            <div class="movie-card" style="border-left: 5px solid gold;">
                <div class="card-title">{row['title']}</div>
                <div class="card-meta">
                    <span style="color:gold;">★ {row['vote_average']}</span>
                </div>
                <div class="plot-preview">{row['overview']}</div>
            </div>
            """, unsafe_allow_html=True)
