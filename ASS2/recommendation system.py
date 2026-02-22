import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df_reviews = pd.read_csv("user_reviews.csv")
df_genres = pd.read_csv("movie_genres.csv")

df_reviews.set_index('User', inplace=True)
df_genres.set_index('movie_title', inplace=True)

common_movies = df_reviews.columns.intersection(df_genres.index)


df_reviews = df_reviews[common_movies]
df_genres = df_genres.loc[common_movies]


def get_user_score(target_user, reviews_df, top_sim_user_number = 20):

    # caculate users similarity
    sim_matrix = cosine_similarity(reviews_df)
    user_sim = pd.Series(sim_matrix[reviews_df.index.get_loc(target_user)], index = reviews_df.index)

    # find top x neighbours
    top_x_users = user_sim.drop(target_user).sort_values(ascending=False)
    top_x_users = top_x_users.head(top_sim_user_number).index

    user_ratings = reviews_df.loc[target_user]
    unseen_movies = user_ratings[user_ratings == 0].index

    pred_scores = {}

    for movie in unseen_movies:

        neighbour_ratings = reviews_df.loc[top_x_users, movie]
        valid_neighbour = neighbour_ratings[neighbour_ratings > 0]

        if len(valid_neighbour) > 0:
            # weighted average prediction
            weights = user_sim.loc[valid_neighbour.index]
            prediction = np.dot(valid_neighbour,weights) / weights.sum()
            pred_scores[movie] = prediction
        else:
            pred_scores[movie] = 0

    return pd.Series(pred_scores)

def get_content_score(target_user, reviews_df, genre_df):

    user_ratings = reviews_df.loc[target_user]

    rated_movies = user_ratings[user_ratings > 2].index

    if len(rated_movies) == 0:
        return pd.Series(0, index=reviews_df.columns)
    
    # build user image
    movies_feature = genre_df.loc[rated_movies]
    weights = user_ratings.loc[rated_movies].values.reshape(-1, 1)
    user_image = (movies_feature * weights).sum(axis = 0) / weights.sum()

    # calculate unseens movies similarity
    unseens_movies = user_ratings [user_ratings == 0].index
    feature = genre_df.loc[unseens_movies]

    sims_matrix = cosine_similarity(user_image.values.reshape(1, -1), feature)

    return pd.Series(sims_matrix[0], index=unseens_movies)

def hybird_recommend_system(target_user, alpha = 0.6):

    print(f"Generating recommendation for {target_user}")

    user_score = get_user_score(target_user,df_reviews)
    content_score = get_content_score(target_user,df_reviews,df_genres)

    # Normalize
    user_score_norm = user_score / 5.0

    final_score = pd.Series(0.0, index=content_score.index)

    for movie in final_score.index:

        final_score[movie] = alpha * user_score_norm.get(movie, 0) + (1 - alpha) * content_score.get(movie, 0)

    # return top 5
    return final_score.sort_values(ascending=False).head(5).index.tolist()
    
user_list = ['Vincent', 'Edgar', 'Addilyn', 'Marlee', 'Javier']
results = {}

for user in user_list:
    if user in df_reviews.index:
        results[user] = hybird_recommend_system(user)
    else:
        print(f"User {user} not found.")

print("Recommendation Results")
for user, movie in results.items():
    print(f"{user}: {movie}")