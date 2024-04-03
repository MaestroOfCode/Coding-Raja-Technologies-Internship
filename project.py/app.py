from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__,)



# Function to load data from CSV files
def load_data(movies_path, credits_path):
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)
    return movies_df, credits_df


# Function to preprocess data and calculate similarity matrix
def preprocess_and_calculate_similarity(movies_df, credits_df):
    # Merge movies and credits data
    movies_df['cast'] = credits_df['cast'].apply(lambda x: [actor['name'] for actor in eval(x)])
    movies_df['crew'] = credits_df['crew'].apply(
        lambda x: [crew_member['name'] for crew_member in eval(x) if crew_member['job'] == 'Director'])

    # Create bag of words from cast and crew lists
    movies_df['cast_str'] = movies_df['cast'].apply(lambda x: ' '.join(x))
    movies_df['crew_str'] = movies_df['crew'].apply(lambda x: ' '.join(x))
    movies_df['bag_of_words'] = movies_df['cast_str'] + ' ' + movies_df['crew_str']

    # Initialize CountVectorizer and fit_transform on bag of words
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(movies_df['bag_of_words'])

    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    return cosine_sim


# Function to recommend movies using collaborative filtering
def recommend_movies(selected_movie, movies_df, cosine_sim):
    # Get index of selected movie
    selected_movie_index = movies_df[movies_df['title'] == selected_movie].index[0]

    # Get similarity scores for selected movie
    similarity_scores = list(enumerate(cosine_sim[selected_movie_index]))

    # Sort movies based on similarity scores
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Extract top 5 similar movies (excluding the selected movie itself)
    recommended_movies = [(movies_df.iloc[index]['title'], similarity) for index, similarity in
                          sorted_similar_movies[1:6]]

    return recommended_movies


@app.route('/')
def index():
    movies_df, credits_df = load_data('data/tmdb_5000_movies.csv', 'data/tmdb_5000_credits.csv')
    available_movies = movies_df['title'].tolist()
    return render_template('index.html', available_movies=available_movies)


@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form['selected_movie']
    movies_df, credits_df = load_data('data/tmdb_5000_movies.csv', 'data/tmdb_5000_credits.csv')
    cosine_sim = preprocess_and_calculate_similarity(movies_df, credits_df)
    recommended_movies = recommend_movies(selected_movie, movies_df, cosine_sim)
    return render_template('recommendation.html', recommended_movies=recommended_movies)


if __name__ == '__main__':
    app.run(debug=True)
