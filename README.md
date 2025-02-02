Movie Recommendation System

Overview

This is a Flask-based Movie Recommendation System that provides recommendations based on content-based filtering and collaborative filtering using the Expectation-Maximization (EM) Algorithm (Gaussian Mixture Model, GMM). The web application features a sci-fi neon-themed UI and allows users to input a preferred genre and year to receive personalized recommendations.

Features

✅ Hybrid Recommendation System (Content-Based + Collaborative Filtering)✅ Expectation-Maximization Clustering (Gaussian Mixture Model)✅ Sci-Fi Themed User Interface (Neon Colors & Dark UI)✅ Similarity Score for Each Recommendation✅ User-Specified Year Restriction (Movies recommended only up to the input year)

Technologies Used

Component

Technology

Backend (Server)

Flask (Python)

Data Processing

Pandas, NumPy

Machine Learning

Scikit-learn (Gaussian Mixture Model, TF-IDF, Cosine Similarity)

Frontend (UI)

HTML, CSS (Sci-Fi Neon Theme)

Data Storage

CSV (movies.csv)

Folder Structure

project-folder/
├── app.py                     # Main Flask application
├── sci_fi_movies.csv          # Movie data (50 entries)
├── templates/
│   ├── index.html              # Home Page
│   ├── recommendations_content.html  # Content-Based Results
│   ├── recommendations_collaborative.html  # Collaborative-Based Results
└── static/
    ├── styles.css              # Sci-Fi Themed UI

Installation and Setup

Step 1: Install Dependencies

Run the following command in your terminal:

pip install flask pandas scikit-learn numpy

Step 2: Organize Files

Ensure your folder structure matches the one provided above.

Step 3: Run the Application

python app.py

Step 4: Open the Web App

Go to: http://127.0.0.1:5000

Enter your favorite genre and year, and select recommendation type.

Usage

Home Page

Enter preferred genre (e.g., Sci-Fi, Romance, Thriller, etc.).

Enter a specific year (Only movies released on or before this year are considered).

Choose between Content-Based or Collaborative-Based recommendations.

Content-Based Filtering

Uses TF-IDF Vectorization to analyze genres and years.

Applies the Expectation-Maximization (GMM) to cluster similar movies.

Ranks movies based on cosine similarity score.

Collaborative Filtering

Uses user ratings to build a user-movie matrix.

Applies the Expectation-Maximization (GMM) for clustering.

Ranks movies based on similar user preferences.

