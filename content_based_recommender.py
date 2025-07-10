import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import difflib

def get_movie_title_from_index(index, movies_df):
    """Helper to get movie title from DataFrame index."""
    # Ensure index is within bounds before accessing
    if 0 <= index < len(movies_df):
        return movies_df.iloc[index]['title']
    return None

def get_index_from_movie_title(title, movies_df):
    """Helper to get DataFrame index from movie title (handles fuzzy matching)."""
    all_movie_titles = movies_df['title'].tolist()
    # Using difflib to find the closest match
    close_matches = difflib.get_close_matches(title, all_movie_titles, n=1, cutoff=0.6)
    if close_matches:
        matched_title = close_matches[0]
        # Return the *index* of the matched title in the DataFrame
        return movies_df[movies_df['title'] == matched_title].index[0]
    return None

if __name__ == "__main__":
    print("--- Content-Based Recommendation System ---")

    # --- Configuration ---
    MOVIES_FILE = 'movies.csv'
    TOP_N_RECOMMENDATIONS = 10

    # --- 1. Load the movies dataset ---
    print(f"Loading {MOVIES_FILE}...")
    try:
        movies_df = pd.read_csv(MOVIES_FILE, low_memory=False)
        # Handle movies with no genres listed by filling with an empty string
        # This is important for TfidfVectorizer as it expects string input
        movies_df['genres'] = movies_df['genres'].fillna('')
        print(f"Successfully loaded {MOVIES_FILE}. Shape: {movies_df.shape}")
    except FileNotFoundError:
        print(f"Error: {MOVIES_FILE} not found. Make sure it's in the same directory as this script.")
        exit()

    # --- 2. Create TF-IDF Vectorizer for Genres ---
    # TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a genre is to a movie.
    # It converts a collection of raw documents (movie genres strings) into a matrix of TF-IDF features.
    print("Vectorizing movie genres using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # Fit and transform the genres data
    # If a movie has '(no genres listed)', TfidfVectorizer will treat it as an empty document, which is fine.
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])
    print(f"TF-IDF matrix created with shape: {tfidf_matrix.shape}")

    # --- 3. Compute Cosine Similarity Matrix ---
    # Cosine similarity measures the cosine of the angle between two vectors.
    # The closer the vectors are in orientation, the larger the cosine similarity.
    # It indicates how similar two movies are based on their genre profiles.
    print("Computing cosine similarity between movies (this may take a moment)...")
    # Using linear_kernel for faster computation of dot product for sparse matrices.
    # For TF-IDF vectors, linear_kernel computes the cosine similarity.
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print("Cosine similarity matrix computed.")

    # --- 4. Define Recommendation Function ---
    def get_content_based_recommendations(movie_title, cosine_sim_matrix, movies_dataframe, top_n=TOP_N_RECOMMENDATIONS):
        # Get the index of the movie that matches the title
        idx = get_index_from_movie_title(movie_title, movies_dataframe)

        if idx is None:
            print(f"Movie '{movie_title}' not found in the database. Please try another title.")
            return []

        # Get the pairwise similarity scores of all movies with that movie
        # Enumerate to keep track of the original index of movies
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))

        # Sort the movies based on the similarity scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top_n most similar movies. Exclude the movie itself (which will have a similarity of 1.0).
        sim_scores = sim_scores[1:top_n+1]

        # Prepare the list of recommended movies with their titles and similarity scores
        recommended_movies = []
        for i, score in sim_scores: # sim_scores now contains (index, score) pairs
            title = get_movie_title_from_index(i, movies_dataframe)
            if title: # Ensure title was found
                recommended_movies.append({'title': title, 'similarity_score': score})
        return recommended_movies

    # --- 5. User Interaction Loop ---
    print("\nType a movie title to get content-based recommendations, or 'quit' to exit.")

    while True:
        user_input_movie_title = input("\nEnter a movie title (e.g., Toy Story, Pulp Fiction, quit to exit): ").strip()
        if user_input_movie_title.lower() == 'quit':
            print("Exiting content-based recommendation system. Goodbye!")
            break

        recommendations = get_content_based_recommendations(user_input_movie_title, cosine_sim, movies_df)

        if recommendations:
            print(f"\nTop {TOP_N_RECOMMENDATIONS} recommendations for movies similar to '{user_input_movie_title}':")
            for rec in recommendations:
                print(f"- {rec['title']} (Similarity: {rec['similarity_score']:.3f})")
        else:
            # Error message is already printed by get_index_from_movie_title
            pass