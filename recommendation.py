# ## Recommendation System

# %%
tech_cols = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "DevEnvsHaveWorkedWith",
    "AIModelsHaveWorkedWith",
]

survey[tech_cols].head()

# %%
[
    col for col in tech_cols if col in survey.columns
]  # Checking if columns exist in dataframe

# %%
language_matrix = survey["LanguageHaveWorkedWith"].fillna("").str.get_dummies(sep=";")

language_matrix.head()

# %%
database_matrix = survey["DatabaseHaveWorkedWith"].fillna("").str.get_dummies(sep=";")

database_matrix.head()

# %%
webframe_matrix = survey["WebframeHaveWorkedWith"].fillna("").str.get_dummies(sep=";")

webframe_matrix.head()

# %%
platform_matrix = survey["PlatformHaveWorkedWith"].fillna("").str.get_dummies(sep=";")

platform_matrix.head()

# %%
devs_matrix = survey["DevEnvsHaveWorkedWith"].fillna("").str.get_dummies(sep=";")

devs_matrix.head()

# %%
ai_matrix = survey["AIModelsHaveWorkedWith"].fillna("").str.get_dummies(sep=";")

ai_matrix.head()

# %%
print("Languages:", language_matrix.shape)
print("Databases:", database_matrix.shape)
print("DevEnvs:", devs_matrix.shape)
print("Web Frameworks:", webframe_matrix.shape)
print("Platforms:", platform_matrix.shape)
print("AI:", ai_matrix.shape)

# %%
technology_matrix = pd.concat(
    [
        language_matrix,
        database_matrix,
        devs_matrix,
        webframe_matrix,
        platform_matrix,
        ai_matrix,
    ],
    axis=1,
)

# %%
technology_matrix.shape

# %%
technology_matrix.head()

# %%
language_matrix = language_matrix.add_prefix("lang_")
database_matrix = database_matrix.add_prefix("db_")
devs_matrix = devs_matrix.add_prefix("dev_")
webframe_matrix = webframe_matrix.add_prefix("web_")
platform_matrix = platform_matrix.add_prefix("platform_")
ai_matrix = ai_matrix.add_prefix("ai_")

# Giving prefix so that there won't be duplicate

# %%
technology_matrix = pd.concat(
    [
        language_matrix,
        database_matrix,
        devs_matrix,
        webframe_matrix,
        platform_matrix,
        ai_matrix,
    ],
    axis=1,
)

# %%
technology_matrix.head()

# %%
technology_matrix.isna().sum().sum()

# %%
technology_matrix.dtypes.value_counts()

# %%
technology_matrix.to_csv("scaled.csv", index=False)

# %%
technology_matrix = pd.read_csv("scaled.csv")

# %%
technology_matrix["technology_count"] = technology_matrix.sum(axis=1)

# %%
X_technology = technology_matrix

# %%
technology_count = X_technology.sum(axis=1)

# %%
developer_0 = X_technology.iloc[0]

developer_0[developer_0 == 1].index.tolist()

# %%
print("Technology matrix shape:", X_technology.shape)
print("Average technologies:", technology_count.mean())
print("Maximum technologies:", technology_count.max())

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ### Function for finding similar developers

# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_similar_developers(developer_index, X_technology, n=10):
    # Target developer
    target = X_technology.iloc[[developer_index]]

    # Compare target with every developer
    similarities = cosine_similarity(target, X_technology)[0]

    # Sort from highest similarity to lowest
    similar_indices = np.argsort(similarities)[::-1]

    # Remove the target developer itself
    similar_indices = [i for i in similar_indices if i != developer_index]

    # Keep top N
    similar_indices = similar_indices[:n]

    # Create result
    result = pd.DataFrame(
        {
            "developer_index": similar_indices,
            "similarity": similarities[similar_indices],
        }
    )

    return result


# %%
similar_developers = find_similar_developers(
    developer_index=11, X_technology=X_technology, n=5
)

similar_developers

# %%
target_profile = X_technology.iloc[0]

current_technologies = target_profile[target_profile == 1].index.tolist()

current_technologies

# %%
similar_index = similar_developers.iloc[0]["developer_index"]

similar_index = int(similar_index)

similar_profile = X_technology.iloc[similar_index]

similar_technologies = similar_profile[similar_profile == 1].index.tolist()

similar_technologies

# %%
current_technologies = set(X_technology.iloc[0][X_technology.iloc[0] == 1].index)

similar_technologies = set(
    X_technology.iloc[similar_index][X_technology.iloc[similar_index] == 1].index
)

new_technologies = similar_technologies - current_technologies

new_technologies

# %% [markdown]
# ### Function for recommending things to learn to developers based on what developers similar to them have learnt

# %%
from collections import Counter


def recommend_technologies(
    developer_index, X_technology, n_similar=20, n_recommendations=5
):
    # --------------------------------
    # Step 1: Find similar developers
    # --------------------------------

    similar_df = find_similar_developers(developer_index, X_technology, n=n_similar)

    # --------------------------------
    # Step 2: Technologies already known
    # --------------------------------

    target_profile = X_technology.iloc[developer_index]

    current_technologies = set(target_profile[target_profile == 1].index)

    # --------------------------------
    # Step 3: Collect technologies
    # --------------------------------

    recommendations = Counter()

    for _, row in similar_df.iterrows():
        idx = int(row["developer_index"])
        similarity = row["similarity"]

        profile = X_technology.iloc[idx]

        technologies = profile[profile == 1].index

        for tech in technologies:
            # Don't recommend something
            # the developer already has
            if tech not in current_technologies:
                recommendations[tech] += similarity

    # --------------------------------
    # Step 4: Rank recommendations
    # --------------------------------

    ranked = recommendations.most_common(n_recommendations)

    return pd.DataFrame(ranked, columns=["technology", "recommendation_score"])


# %%
recommendations = recommend_technologies(
    developer_index=426, X_technology=X_technology, n_similar=20, n_recommendations=10
)

recommendations
