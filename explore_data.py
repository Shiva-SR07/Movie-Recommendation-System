import pandas as pd

# --- Configuration ---
MOVIES_FILE = 'movies.csv'
RATINGS_FILE = 'ratings.csv'

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

# --- 2. Inspect the first few rows (head) ---
print("\n--- Movies DataFrame Head ---")
print(movies_df.head())

print("\n--- Ratings DataFrame Head ---")
print(ratings_df.head())

# --- 3. Get general information about the DataFrames (datatypes, non-null counts) ---
print("\n--- Movies DataFrame Info ---")
movies_df.info()

print("\n--- Ratings DataFrame Info ---")
ratings_df.info()

# --- 4. Get basic descriptive statistics ---
print("\n--- Movies DataFrame Description ---")
print(movies_df.describe())

print("\n--- Ratings DataFrame Description ---")
print(ratings_df.describe())

# --- 5. Custom Exploratory Data Analysis (EDA) ---

# Number of unique users and movies
num_users = ratings_df['userId'].nunique()
num_movies = movies_df['movieId'].nunique()
total_ratings = ratings_df.shape[0]

print(f"\nTotal unique users: {num_users}")
print(f"Total unique movies in movies.csv: {num_movies}")
print(f"Total ratings recorded: {total_ratings}")

# Average rating
average_rating = ratings_df['rating'].mean()
print(f"Overall average rating: {average_rating:.2f}")

# Distribution of ratings
print("\nDistribution of ratings:")
print(ratings_df['rating'].value_counts().sort_index())

# Most rated movies
print("\nTop 10 most rated movies:")
# Count ratings per movie, sort, merge with movie titles
movie_rating_counts = ratings_df['movieId'].value_counts().reset_index()
movie_rating_counts.columns = ['movieId', 'num_ratings']
top_movies_by_ratings = pd.merge(movie_rating_counts, movies_df, on='movieId', how='left')
print(top_movies_by_ratings.head(10))

# Top-rated movies (considering minimum number of ratings)
print("\nTop 10 movies by average rating (minimum 500 ratings):")
min_ratings_threshold = 500 # Adjust this based on your dataset size

# Calculate average rating and number of ratings for each movie
movie_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.columns = ['movieId', 'average_rating', 'num_ratings']

# Filter for movies with enough ratings
popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings_threshold]

# Merge with movie titles and sort by average rating
top_rated_popular_movies = pd.merge(popular_movies, movies_df, on='movieId', how='left')
top_rated_popular_movies = top_rated_popular_movies.sort_values(by='average_rating', ascending=False)
print(top_rated_popular_movies.head(10))

print("\n--- Data Exploration Complete ---")