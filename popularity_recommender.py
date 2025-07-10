import pandas as pd

# --- Configuration ---
MOVIES_FILE = 'movies.csv'
RATINGS_FILE = 'ratings.csv'
TOP_N = 20 # Number of recommendations to display
MIN_RATINGS_THRESHOLD = 500 # Minimum ratings a movie needs to be considered for average rating calculation

# --- 1. Load the datasets ---
print(f"Loading {MOVIES_FILE}...")
try:
    movies_df = pd.read_csv(MOVIES_FILE, low_memory=False)
    print(f"Successfully loaded {MOVIES_FILE}. Shape: {movies_df.shape}")
except FileNotFoundError:
    print(f"Error: {MOVIES_FILE} not found. Make sure it's in the same directory as this script.")
    exit()

print(f"\nLoading {RATINGS_FILE} (this might take a moment for large files)...")
try:
    ratings_df = pd.read_csv(RATINGS_FILE, low_memory=False)
    print(f"Successfully loaded {RATINGS_FILE}. Shape: {ratings_df.shape}")
except FileNotFoundError:
    print(f"Error: {RATINGS_FILE} not found. Make sure it's in the same directory as this script.")
    exit()

# --- 2. Implement Popularity-Based Recommender ---

print(f"\n--- Popularity-Based Recommendations (Top {TOP_N}) ---")

# Calculate number of ratings for each movie
print("\nCalculating number of ratings for each movie...")
movie_rating_counts = ratings_df['movieId'].value_counts().reset_index()
movie_rating_counts.columns = ['movieId', 'num_ratings']

# Merge with movies_df to get titles
top_movies_by_ratings = pd.merge(movie_rating_counts, movies_df, on='movieId', how='left')
print(f"\nTop {TOP_N} Movies by Number of Ratings:")
print(top_movies_by_ratings.head(TOP_N))


# Calculate average rating and number of ratings for each movie
print(f"\nCalculating average ratings for movies with at least {MIN_RATINGS_THRESHOLD} ratings...")
movie_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.columns = ['movieId', 'average_rating', 'num_ratings']

# Filter for movies with enough ratings
popular_movies_filtered = movie_stats[movie_stats['num_ratings'] >= MIN_RATINGS_THRESHOLD]

# Merge with movie titles and sort by average rating
top_rated_popular_movies = pd.merge(popular_movies_filtered, movies_df, on='movieId', how='left')
top_rated_popular_movies = top_rated_popular_movies.sort_values(by='average_rating', ascending=False)

print(f"\nTop {TOP_N} Movies by Average Rating (min {MIN_RATINGS_THRESHOLD} ratings):")
print(top_rated_popular_movies.head(TOP_N))

print("\n--- Popularity Recommender Complete ---")