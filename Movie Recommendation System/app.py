from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movies data
movies_df = pd.read_csv("movies.csv")  # Ensure this file has columns 'title', 'genres', 'year', 'user_id', 'user_ratings'
movies_df['year'] = movies_df['year'].astype(int)

def preprocess_content_features(movies_df):
    movies_df['features'] = movies_df['genres'] + " " + movies_df['year'].astype(str)
    vectorizer = TfidfVectorizer()
    content_vectors = vectorizer.fit_transform(movies_df['features'])
    return content_vectors

def get_em_recommendations_content(movies_df, content_vectors, genres_input, year_input, n_components=5):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(content_vectors.toarray())
    movies_df['cluster_content'] = gmm.predict(content_vectors.toarray())

    matching_movies = movies_df[
        (movies_df['genres'].str.contains(genres_input, case=False, na=False)) & 
        (movies_df['year'] <= year_input)
    ]
    
    if matching_movies.empty:
        return pd.DataFrame()
    
    matching_clusters = matching_movies['cluster_content'].unique()
    recommendations = movies_df[movies_df['cluster_content'].isin(matching_clusters)]
    recommendations['similarity_score'] = cosine_similarity(
        content_vectors[matching_movies.index, :], content_vectors[recommendations.index, :]).mean(axis=0)
    
    return recommendations[['title', 'genres', 'year', 'similarity_score']].sort_values(by='similarity_score', ascending=False).head(5)

def get_em_recommendations_collaborative(movies_df, user_id, n_components=5):
    # Create a pivot table to fill in missing user ratings with 0
    ratings_matrix = movies_df.pivot_table(index='title', columns='user_id', values='user_ratings').fillna(0)
    
    # Check if user_id is in the matrix
    if user_id not in ratings_matrix.columns:
        return pd.DataFrame()  # If user_id not found, return an empty DataFrame
    
    # Fit a Gaussian Mixture Model for clustering
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(ratings_matrix.values)
    movies_df['cluster_collaborative'] = gmm.predict(ratings_matrix.values)
    
    # Find the user’s cluster
    user_cluster = movies_df[movies_df['user_id'] == user_id]['cluster_collaborative'].iloc[0]
    recommendations = movies_df[movies_df['cluster_collaborative'] == user_cluster]
    
    # Align vectors for cosine similarity
    user_vector = ratings_matrix[user_id].values.reshape(1, -1)  # Vector of the target user's ratings
    
    # Get recommendation vectors
    recommendation_vectors = ratings_matrix.loc[recommendations['title']].values  # Vector for recommendations
    
    # Identify common columns
    common_columns = ratings_matrix.columns.intersection(recommendations['user_id']).to_list()
    
    # Get indices for common columns
    user_vector_common = user_vector[:, ratings_matrix.columns.get_indexer(common_columns)]
    recommendation_vectors_common = recommendation_vectors[:, ratings_matrix.columns.get_indexer(common_columns)]
    
    # Calculate similarity scores
    recommendations['similarity_score'] = cosine_similarity(user_vector_common, recommendation_vectors_common).flatten()
    
    return recommendations[['title', 'genres', 'year', 'similarity_score']].sort_values(by='similarity_score', ascending=False).head(5)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend_content', methods=['POST'])
def recommend_content():
    genres_input = request.form['genres']
    year_input = int(request.form['year'])
    
    content_vectors = preprocess_content_features(movies_df)
    recommendations = get_em_recommendations_content(movies_df, content_vectors, genres_input, year_input)
    
    return render_template('recommendations_content.html', movies=recommendations)

@app.route('/recommend_collaborative', methods=['POST'])
def recommend_collaborative():
    user_id = int(request.form['user_id'])  # Get user ID from form
    recommendations = get_em_recommendations_collaborative(movies_df, user_id)
    return render_template('recommendations_collaborative.html', movies=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
