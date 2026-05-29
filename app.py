import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Netflix AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# TMDB API
# ---------------------------------------------------

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = st.secrets.get("TMDB_API_KEY") or os.environ.get("TMDB_API_KEY")

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0b0b0b;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0b0b0b;
}

/* HERO SECTION */

.hero {
    text-align: center;
    padding-top: 20px;
    padding-bottom: 30px;
}

.hero-title {
    font-size: 70px;
    font-weight: 900;
    color: #E50914;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 22px;
    color: #bbbbbb;
    margin-bottom: 20px;
}

/* SELECT BOX */

div[data-baseweb="select"] {
    background-color: #1c1c1c !important;
    border-radius: 12px !important;
    color: white !important;
}

/* BUTTON */

.stButton>button {
    background: linear-gradient(90deg, #E50914, #ff1f1f);
    color: white;
    border: none;
    border-radius: 12px;
    height: 60px;
    width: 280px;
    font-size: 22px;
    font-weight: bold;
    transition: 0.3s;
    box-shadow: 0px 4px 20px rgba(229,9,20,0.5);
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff1f1f, #E50914);
}

/* MOVIE CARD */

.movie-card {
    background-color: #141414;
    border-radius: 18px;
    overflow: hidden;
    padding-bottom: 15px;
    transition: 0.4s;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.6);
    margin-bottom: 25px;
}

.movie-card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0px 10px 30px rgba(229,9,20,0.4);
}

.movie-title {
    font-size: 24px;
    font-weight: bold;
    padding: 10px;
    color: white;
}

.movie-rating {
    color: gold;
    font-size: 18px;
    padding-left: 10px;
}

.movie-date {
    color: #bbbbbb;
    font-size: 15px;
    padding-left: 10px;
}

.movie-overview {
    color: #dddddd;
    font-size: 14px;
    padding: 10px;
    height: 120px;
    overflow: hidden;
}

/* IMAGE */

img {
    border-radius: 15px 15px 0px 0px;
}

/* SCROLLBAR */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #111;
}

::-webkit-scrollbar-thumb {
    background: #E50914;
    border-radius: 10px;
}

/* MOBILE */

@media screen and (max-width: 768px) {

    .hero-title {
        font-size: 42px;
    }

    .hero-subtitle {
        font-size: 16px;
    }

    .stButton>button {
        width: 100%;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

movies = movies.merge(credits, on='title')

movies = movies[
    [
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew'
    ]
]

movies.dropna(inplace=True)

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def convert(text):

    result = []

    for item in ast.literal_eval(text):
        result.append(item['name'])

    return result


def convert_cast(text):

    result = []
    counter = 0

    for item in ast.literal_eval(text):

        if counter != 3:
            result.append(item['name'])
            counter += 1
        else:
            break

    return result


def fetch_director(text):

    result = []

    for item in ast.literal_eval(text):

        if item['job'] == 'Director':
            result.append(item['name'])

    return result

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['tags'] = movies['overview'] + \
                 movies['genres'] + \
                 movies['keywords'] + \
                 movies['cast'] + \
                 movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# ---------------------------------------------------
# VECTORIZE
# ---------------------------------------------------

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# ---------------------------------------------------
# FETCH MOVIE DETAILS
# ---------------------------------------------------

def fetch_movie_details(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"

    response = requests.get(url)

    data = response.json()

    poster_path = data.get('poster_path')

    if poster_path:
        poster = "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        poster = "https://via.placeholder.com/500x750?text=No+Image"

    rating = data.get('vote_average', "N/A")

    release_date = data.get('release_date', "Unknown")

    overview = data.get('overview', "No overview available.")

    return poster, rating, release_date, overview

# ---------------------------------------------------
# RECOMMEND FUNCTION
# ---------------------------------------------------

def recommend(movie):

    movie = movie.lower()

    movie_index = new_df[
        new_df['title'].str.lower() == movie
    ].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []
    recommended_ratings = []
    recommended_dates = []
    recommended_overviews = []

    for i in movies_list:

        movie_id = new_df.iloc[i[0]].movie_id

        recommended_movies.append(
            new_df.iloc[i[0]].title
        )

        poster, rating, release_date, overview = fetch_movie_details(movie_id)

        recommended_posters.append(poster)
        recommended_ratings.append(rating)
        recommended_dates.append(release_date)
        recommended_overviews.append(overview)

    return (
        recommended_movies,
        recommended_posters,
        recommended_ratings,
        recommended_dates,
        recommended_overviews
    )

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-title">
        🎬 Netflix AI Recommender
    </div>

    <div class="hero-subtitle">
        Discover movies powered by Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MOVIE SELECT
# ---------------------------------------------------

movie_list = new_df['title'].values

selected_movie = st.selectbox(
    "🎥 Choose your favorite movie",
    movie_list
)

st.write("")

# ---------------------------------------------------
# RECOMMEND BUTTON
# ---------------------------------------------------

if st.button("🔥 Recommend Movies"):

    with st.spinner("Finding amazing movies for you..."):

        (
            names,
            posters,
            ratings,
            dates,
            overviews
        ) = recommend(selected_movie)

    st.write("")
    st.subheader("🍿 Recommended For You")

    cols = st.columns(5)

    for i in range(5):

        with cols[i]:

            st.markdown(
                """
                <div class="movie-card">
                """,
                unsafe_allow_html=True
            )

            st.image(posters[i])

            st.markdown(
                f"""
                <div class="movie-title">
                    {names[i]}
                </div>

                <div class="movie-rating">
                    ⭐ {ratings[i]}
                </div>

                <div class="movie-date">
                    📅 {dates[i]}
                </div>

                <div class="movie-overview">
                    {overviews[i][:140]}...
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )
