
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample dataset
movies = {
    "title": [
        "The Dark Knight",
        "Batman Begins",
        "The Dark Knight Rises",
        "Iron Man",
        "Avengers Endgame",
        "Avengers Infinity War",
        "Captain America Civil War",
        "Thor Ragnarok",
        "Interstellar",
        "Inception"
    ],
    "genre": [
        "action crime drama",
        "action crime thriller",
        "action thriller drama",
        "action sci-fi superhero",
        "action superhero sci-fi",
        "action superhero sci-fi",
        "action superhero drama",
        "action comedy superhero",
        "sci-fi drama space",
        "sci-fi thriller dream"
    ]
}

df = pd.DataFrame(movies)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["genre"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(movie_title, top_n=5):
    if movie_title not in df["title"].values:
        return []

    idx = df[df["title"] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_movies = scores[1:top_n+1]
    return [df["title"][i[0]] for i in top_movies]

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    selected = None

    if request.method == "POST":
        selected = request.form.get("movie")
        recommendations = recommend(selected)

    return render_template("index.html",
                           movies=df["title"].tolist(),
                           recommendations=recommendations,
                           selected=selected)

if __name__ == "__main__":
    app.run(debug=True)
