import pandas as pd
from pathlib import Path

def find_pattern_movie(reviews_df):
    """Find the movie with largest number of reviews following same score pattern."""
    print("Finding movie with consistent score pattern...")
    
    # Pre-filter for /5 and /10 patterns only
    pattern_5_reviews = reviews_df[reviews_df['originalScore'].str.contains('/5', na=False)]
    pattern_10_reviews = reviews_df[reviews_df['originalScore'].str.contains('/10', na=False)]
    
    best_movie = None
    best_pattern = None
    best_score = 0
    
    # Check /5 pattern
    if len(pattern_5_reviews) > 0:
        movie_counts = pattern_5_reviews.groupby('id').size()
        for movie_id, count in movie_counts.items():
            if count >= 50:  # Need at least 50 reviews
                unique_scores = pattern_5_reviews[pattern_5_reviews['id'] == movie_id]['originalScore'].nunique()
                score = count * unique_scores  # Prefer more reviews with more diversity
                if score > best_score:
                    best_movie = movie_id
                    best_pattern = '/5'
                    best_score = score
    
    # Check /10 pattern
    if len(pattern_10_reviews) > 0:
        movie_counts = pattern_10_reviews.groupby('id').size()
        for movie_id, count in movie_counts.items():
            if count >= 50:  # Need at least 50 reviews
                unique_scores = pattern_10_reviews[pattern_10_reviews['id'] == movie_id]['originalScore'].nunique()
                score = count * unique_scores
                if score > best_score:
                    best_movie = movie_id
                    best_pattern = '/10'
                    best_score = score
    
    print(f"Selected pattern movie: {best_movie} with {best_pattern} pattern")
    return best_movie, best_pattern


def get_negative_movie():
    """Use hardcoded negative movie."""
    print("Using hardcoded negative movie: taken_3")
    return 'taken_3'


def get_top_movies_fast(reviews_df, top_n=200):
    """Get top movies by review count quickly."""
    print(f"Getting top {top_n} movies by review count...")
    movie_counts = reviews_df.groupby('id').size().sort_values(ascending=False)
    return movie_counts.head(top_n).index.tolist()


def sample_reviews(reviews_df, pattern_movie, pattern_pattern, negative_movie, top_movies, scale_factor):
    """Sample reviews using movie-first strategy."""
    print(f"Sampling {scale_factor} reviews using movie-first strategy...")
    
    selected_reviews = []
    used_movies = set()
    
    # Step 1: Add pattern movie reviews (ONLY those matching the pattern)
    print(f"Step 1: Adding pattern movie {pattern_movie} (only {pattern_pattern} reviews)")
    pattern_movie_reviews = reviews_df[
        (reviews_df['id'] == pattern_movie) & 
        (reviews_df['originalScore'].str.contains(pattern_pattern, na=False))
    ]
    selected_reviews.append(pattern_movie_reviews)
    used_movies.add(pattern_movie)
    remaining = scale_factor - len(pattern_movie_reviews)
    print(f"Added {len(pattern_movie_reviews)} pattern reviews, remaining: {remaining}")
    
    # Step 2: Add negative movie reviews (ALL reviews, no pattern filtering)
    if negative_movie != pattern_movie and remaining > 0:
        print(f"Step 2: Adding negative movie {negative_movie} (all reviews)")
        negative_movie_reviews = reviews_df[reviews_df['id'] == negative_movie]
        if len(negative_movie_reviews) > 0:
            sample_size = min(len(negative_movie_reviews), remaining // 3)  # Use up to 1/3 of remaining
            sampled = negative_movie_reviews.sample(n=sample_size, random_state=42)
            selected_reviews.append(sampled)
            used_movies.add(negative_movie)
            remaining -= sample_size
            print(f"Added {sample_size} negative reviews, remaining: {remaining}")
    
    # Step 3: Add reviews from top movies (ALL reviews, no pattern filtering)
    print("Step 3: Adding reviews from top movies (all reviews)")
    for movie_id in top_movies:
        if remaining <= 0:
            break
        if movie_id in used_movies:
            continue
            
        movie_reviews = reviews_df[reviews_df['id'] == movie_id]
        if len(movie_reviews) == 0:
            continue
            
        # Sample 5-15 reviews per movie for diversity
        sample_size = min(len(movie_reviews), max(5, remaining // 30), remaining)
        sampled = movie_reviews.sample(n=sample_size, random_state=42)
        selected_reviews.append(sampled)
        used_movies.add(movie_id)
        remaining -= sample_size
    
    print(f"Final step: added reviews from {len(used_movies)} total movies")
    
    # Combine and shuffle
    final_reviews = pd.concat(selected_reviews, ignore_index=True)
    final_reviews = final_reviews.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Trim to exact scale factor
    final_reviews = final_reviews.head(scale_factor)
    
    return final_reviews


def generate_movies_table(movies_df, reviews_df):
    """Generate movies table for movies that have reviews."""
    movie_ids_with_reviews = reviews_df['id'].unique()
    selected_movies = movies_df[movies_df['id'].isin(movie_ids_with_reviews)].copy()
    return selected_movies



if __name__ == "__main__":

    dataset_path = Path(__file__).parent.parent.parent / "files/movie/data"

    base_table_path = dataset_path / "rotten_tomatoes_movies.csv"
    review_table_path = dataset_path / "rotten_tomatoes_movie_reviews.csv"

    movies_df = pd.read_csv(base_table_path).reset_index(drop=True)
    reviews_df = pd.read_csv(review_table_path).reset_index(drop=True)


    # Find special movies
    pattern_movie, pattern_pattern = find_pattern_movie(reviews_df)
    negative_movie = get_negative_movie()

    # Get top movies
    top_movies = get_top_movies_fast(reviews_df)

    # Sample reviews
    scale_factor = 2000
    selected_reviews = sample_reviews(
        reviews_df,
        pattern_movie,
        pattern_pattern,
        negative_movie,
        top_movies,
        scale_factor=scale_factor
    )

    # Generate movies table
    selected_movies = generate_movies_table(movies_df, selected_reviews)

    selected_reviews["reviewText"] = (
        selected_reviews["reviewText"]
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
    )


    merged_df = selected_movies.merge(
        selected_reviews,
        on="id",
        how="left",
        suffixes=("", "_review")
    )

    merged_df.to_csv(Path(__file__).parent / "data_full.csv", index=False)

    # Update the ground truth files.
    gt_q1 = pd.read_csv("ground_truth/Q1.csv")
    gt_q2 = pd.read_csv("ground_truth/Q2.csv")

    gt_q1 = gt_q1[gt_q1["reviewId"].isin(merged_df["reviewId"])]
    gt_q2 = gt_q2[gt_q2["reviewId"].isin(merged_df["reviewId"])]

    gt_q1.to_csv("ground_truth/Q1.csv", index=False)
    gt_q2.to_csv("ground_truth/Q2.csv", index=False)
