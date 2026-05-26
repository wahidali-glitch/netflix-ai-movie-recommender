import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')

# Select important columns
movies = movies[['movie_id',
                 'title',
                 'overview',
                 'genres',
                 'keywords',
                 'cast',
                 'crew']]

# Remove missing values
movies.dropna(inplace=True)


# Convert genres and keywords
def convert(text):
    result = []

    for item in ast.literal_eval(text):
        result.append(item['name'])

    return result


# Convert cast (top 3 actors)
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


# Fetch director name
def fetch_director(text):
    result = []

    for item in ast.literal_eval(text):

        if item['job'] == 'Director':
            result.append(item['name'])

    return result


# Apply preprocessing
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview into list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces between words
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

# Create tags
movies['tags'] = movies['overview'] + \
                 movies['genres'] + \
                 movies['keywords'] + \
                 movies['cast'] + \
                 movies['crew']

# Create new dataframe
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list into string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Lowercase conversion
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)


# Recommendation function
def recommend(movie):

    movie = movie.lower()

    movie_index = new_df[
        new_df['title'].str.lower() == movie
    ].index

    if len(movie_index) == 0:
        print("\nMovie not found!")
        return

    movie_index = movie_index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\n🎬 Recommended Movies:\n")

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# User input
movie_name = input("Enter movie name: ")

recommend(movie_name)