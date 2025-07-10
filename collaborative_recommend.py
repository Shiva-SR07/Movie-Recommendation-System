import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from collections import defaultdict
import difflib # For fuzzy string matching

def get_movie_name_from_id(movie_id, movies_df):
    """Helper to get movie title from movieId."""
    # Ensure movie_id is treated as numeric for comparison
    return movies_df[movies_df['movieId'] == int(movie_id)]['title'].iloc[0]

def get_id_from_movie_name(movie_title, movies_df):
    """Helper to get movieId from movie title (handles fuzzy matching)."""
    all_movie_titles = movies_df['title'].tolist()
    close_matches = difflib.get_close_matches(movie_title, all_movie_titles, n=1, cutoff=0.6)
    if close_matches:
        matched_title = close_matches[0]
        return movies_df[movies_df['title'] == matched_title]['movieId'].iloc[0]
    return None

def get_top_n_recommendations(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions."""
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

if __name__ == "__main__":
    print("Loading movie data (for titles)...")
    # Using low_memory=False to avoid DtypeWarning for large CSVs
    movies_df = pd.read_csv('movies.csv', low_memory=False)

    print("Loading ratings data (this may take a while for ml-32m)...")
    reader = Reader(rating_scale=(0.5, 5))
    
    # Using low_memory=False for ratings.csv too, given its size
    ratings_df = pd.read_csv('ratings.csv', low_memory=False)

    # Drop the 'timestamp' column as it's not needed for the model
    ratings_df = ratings_df[['userId', 'movieId', 'rating']]

    # Load data from the pandas DataFrame into Surprise Dataset format
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    print("Splitting data into training and testing sets...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    print("Training the collaborative filtering model (KNNBasic - User-Based)...")
    # You could try SVD() for a matrix factorization approach, which is often more accurate
    # algo = SVD(random_state=42, n_epochs=20, lr_all=0.005, reg_all=0.02)
    
    # KNNBasic with user-based cosine similarity
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }
    algo = KNNBasic(sim_options=sim_options)

    algo.fit(trainset)
    print("Model training complete.")

    print("\n--- Collaborative Filtering Recommendation System ---")
    print("Type a movie title (or part of it) that you know a user has rated, or 'quit' to exit.")
    print("We will then try to recommend movies for a 'hypothetical' user who liked that movie.")

    while True:
        user_input_movie_title = input("\nEnter a movie title you want to base recommendations on (e.g., Toy Story, Pulp Fiction, quit to exit): ")
        if user_input_movie_title.lower() == 'quit':
            print("Exiting collaborative recommendation system. Goodbye!")
            break

        user_input_movie_title = user_input_movie_title.strip() # Clean input

        # Find the movieId for the input movie
        input_movie_id = get_id_from_movie_name(user_input_movie_title, movies_df)

        if input_movie_id is None:
            print(f"Movie '{user_input_movie_title}' not found in the database. Please try another title.")
            continue

        # Find a user who has rated this movie highly (e.g., 4 or 5 stars)
        potential_users = ratings_df[(ratings_df['movieId'] == input_movie_id) & (ratings_df['rating'] >= 4.0)]
        
        if potential_users.empty:
            print(f"No users found who rated '{user_input_movie_title}' highly (>=4.0). Cannot base recommendations effectively.")
            print("Try a very popular movie like 'Toy Story' or 'Pulp Fiction'.")
            continue
        
        example_user_id = potential_users.iloc[0]['userId']
        print(f"Using example user '{example_user_id}' who rated '{user_input_movie_title}' highly.")
        
        # Get all movie IDs that this example user HAS NOT rated
        user_rated_movie_ids = set(ratings_df[ratings_df['userId'] == example_user_id]['movieId'])
        all_movie_ids = set(movies_df['movieId'])
        movies_to_predict_for = list(all_movie_ids - user_rated_movie_ids)

        # Filter out movies that the model can't make predictions for (e.g., if they weren't in the trainset)
        # This is a robust check, but for a simple demo, we might skip it if it over-filters.
        # For KNNBasic, it can predict for any item in the training set or with neighbors.
        
        # Create a list of (user, item, 0) tuples for prediction (0 is a dummy true_rating)
        unrated_predictions = []
        for movie_id in movies_to_predict_for:
            unrated_predictions.append((example_user_id, movie_id, 0)) # 0 is a dummy value for true_r

        # Make predictions for unrated movies
        # Filter predictions to only include those where a prediction was actually made (not NaN)
        # and where the estimated rating is not None (Surprise can return None if no prediction can be made)
        predictions_for_user_raw = algo.test(unrated_predictions)
        predictions_for_user = [
            pred for pred in predictions_for_user_raw 
            if pred.est is not None and not pd.isna(pred.est)
        ]


        if not predictions_for_user:
            print(f"Could not make any valid predictions for user {example_user_id}. Try a different movie or user.")
            continue

        # Get top N recommendations for this user
        top_n = get_top_n_recommendations(predictions_for_user, n=10)

        if example_user_id in top_n and top_n[example_user_id]:
            print(f"\nTop 10 recommendations for user (similar to one who liked '{user_input_movie_title}'):")
            for movie_id, estimated_rating in top_n[example_user_id]:
                # Ensure movie_id is treated as integer for lookup
                movie_title = get_movie_name_from_id(int(movie_id), movies_df)
                print(f"- {movie_title} (Estimated Rating: {estimated_rating:.2f})")
        else:
            print(f"Could not generate enough recommendations for user {example_user_id}.")