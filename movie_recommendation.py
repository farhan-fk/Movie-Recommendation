from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
df = pd.read_excel('refined_data_working.xlsx')

import xlrd

# Convert Excel serial numbers to datetime format
df['release_date'] = df['release_date'].apply(lambda x: xlrd.xldate.xldate_as_datetime(x, 0) if not pd.isnull(x) else None)

# Display the first few rows after the correction
# print(df[['original_title', 'release_date']].head())


# Handle missing values
df['runtime'].fillna(df['runtime'].mean(), inplace=True)

# Process categorical columns
for col in ['genres', 'keywords', 'production_companies', 'cast']:
    df[col] = df[col].apply(eval)

# Calculate combined score
df['combined_score'] = df['vote_average'] * (df['vote_count'] / (df['vote_count'] + 1000))

# Convert non-string values in 'over_view' column to empty strings
df['over_view'] = df['over_view'].apply(lambda x: str(x) if isinstance(x, str) else '')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['over_view'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()


# Create Flask app
app = Flask(__name__)

# Flask routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        last_watched_movie = request.form['last_watched_movie']
        recommended_movies = get_recommendations(last_watched_movie, cosine_sim)
        return render_template('recommendations.html', last_watched_movie=last_watched_movie, recommendations=recommended_movies)
    return render_template('recommendations_form.html')

@app.route('/trending')
def trending():
    # Sort movies by popularity in descending order and select only the desired columns
    top_trending_movies = df.sort_values(by='popularity', ascending=False).head(10)[['original_title', 'popularity', 'release_date']]
    print("Top trending movies:", top_trending_movies)
    return render_template('trending.html', trending_movies=top_trending_movies)





@app.route('/most_watched_movie')
def most_watched_movie():
    most_watched = get_most_watched_movie(df)
    return render_template('most_watched_movie.html', most_watched=most_watched)

@app.route('/movies_by_actor', methods=['GET', 'POST'])
def movies_by_actor():
    if request.method == 'POST':
        actor = request.form['actor']
        popular_movies = get_movies_by_actor(df, actor)
        return render_template('movies_by_actor.html', actor=actor, popular_movies=popular_movies)
    return render_template('movies_by_actor_form.html')

# Convert 'director' column to strings to handle NaNs
df['director'] = df['director'].astype(str)

# Extract unique director names and sort them alphabetically
directors = sorted(df['director'].unique())


@app.route('/movies_by_director', methods=['GET', 'POST'])
def movies_by_director():
    if request.method == 'POST':
        director = request.form['director']
        popular_movies = get_movies_by_director(df, director)
        return render_template('movies_by_director.html', director=director, popular_movies=popular_movies, directors=directors)
    return render_template('movies_by_director_form.html', directors=directors)

# Functions for movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['original_title'].iloc[movie_indices].tolist()

def trending_movies(df):
    return df.sort_values(by='popularity', ascending=False)[['original_title', 'popularity', 'release_date']]

def get_most_watched_movie(df):
    return df.sort_values(by='vote_count', ascending=False).iloc[0]['original_title']

def get_movies_by_actor(df, actor):
    return df[df['cast'].apply(lambda x: actor in x)].sort_values(by='popularity', ascending=False)[['original_title', 'popularity', 'release_date']]

def get_movies_by_director(df, director):
    return df[df['director'] == director].sort_values(by='popularity', ascending=False)[['original_title', 'popularity', 'release_date']]

if __name__ == '__main__':
    app.run(debug=True)
