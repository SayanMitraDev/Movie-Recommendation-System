import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
# Dataset should have columns like: title, genres, keywords, overview
df = pd.read_csv("movies.csv")

# Fill missing values
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('')
df['keywords'] = df['keywords'].fillna('')

# Combine important features into one
def combine_features(row):
    return row['overview'] + " " + row['genres'] + " " + row['keywords']

df["combined_features"] = df.apply(combine_features, axis=1)

# Convert text to numbers
vectorizer = CountVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df["combined_features"])

# Compute similarity
similarity = cosine_similarity(feature_matrix)

# Function to get recommendations
def recommend(movie_title):
    if movie_title not in df['title'].values:
        return "Movie not found!"

    movie_index = df[df.title == movie_title].index[0]
    similar_movies = list(enumerate(similarity[movie_index]))

    # Sort by similarity score
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    print(f"\nTop recommendations for '{movie_title}':\n")

    i = 0
    for movie in sorted_movies[1:11]:
        print(df.iloc[movie[0]].title)
        i += 1

# Example
recommend("Avatar")