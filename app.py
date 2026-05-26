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
    page_title="Netflix AI Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# API KEY
# ---------------------------------------------------

API_KEY = "84b0e65085a07a0f29ca92861a347dba"

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0e1117;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0e1117;
}

h1 {
    text-align: center;
    color: #E50914;
    font-size: 65px;
    margin-bottom: 30px;
}

.movie-card {
    background-color: #141414;
    border-radius: 15px;
    padding: 10px;
    transition: transform 0.3s ease;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    height: 100%;
}

.movie-card:hover {
    transform: scale(1.05);
}

.movie-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 10px;
    color: white;
}

.movie-rating {
    color: gold;
    font-size: 16px;
}

.movie-date {
    color: #bbbbbb;
    font-size: 14px;
}

.movie-overview {
    font-size: 13px;
    color: #dddddd;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    height: 55px;
    width: 260px;
    font-size: 20px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #b20710;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

movies = movies.merge(credits, on='title')

movies = movies[[
    'movie_id',
    'title',
    'overview',
    'genres',
    'keywords',
    'cast',
    'crew'
]]

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

    data = requests.get(url)

    data = data.json()

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
# HEADER
# ---------------------------------------------------

st.markdown(
    "<h1>🎬 Netflix AI Movie Recommender</h1>",
    unsafe_allow_html=True
)

st.write("")

# ---------------------------------------------------
# MOVIE SELECT
# ---------------------------------------------------

movie_list = new_df['title'].values

selected_movie = st.selectbox(
    "🎥 Select a movie",
    movie_list
)

st.write("")

# ---------------------------------------------------
# RECOMMEND BUTTON
# ---------------------------------------------------

if st.button("🔥 Recommend Movies"):

    (
        names,
        posters,
        ratings,
        dates,
        overviews
    ) = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):

        with cols[i]:

            st.markdown(
                f"""
                <div class="movie-card">
                """,
                unsafe_allow_html=True
            )

            st.image(posters[i])

            st.markdown(
                f"""
                <div class="movie-title">
                🎬 {names[i]}
                </div>

                <div class="movie-rating">
                ⭐ Rating: {ratings[i]}
                </div>

                <div class="movie-date">
                📅 Release: {dates[i]}
                </div>

                <div class="movie-overview">
                📝 {overviews[i][:120]}...
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)