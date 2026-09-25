# NAME: Jameson Teeters | NUMBER: 5659

import polars as pl
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier

project_folder = Path(__file__).parent

train_df = pl.read_csv(project_folder / "training.csv.gz")
test_df = pl.read_csv(project_folder / "testing.csv.gz")

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

train_df = train_df.with_columns(
    pl.col("outcome").cast(pl.Int8)
)


def add_defender_tracking_features(df):
    df = df.with_columns(
        pl.col("closestdefapproach")
        .str.strip_chars("{}[]")
        .str.split(",")
        .list.eval(
            pl.element()
            .str.strip_chars()
            .cast(pl.Float32)
        )
        .alias("defender_distances")
    )

    df = df.with_columns(
        pl.col("defender_distances").list.get(0).alias("def_dist_1s"),
        pl.col("defender_distances").list.get(1).alias("def_dist_075s"),
        pl.col("defender_distances").list.get(2).alias("def_dist_05s"),
        pl.col("defender_distances").list.get(3).alias("def_dist_025s")
    )

    return df


train_df = add_defender_tracking_features(train_df)
test_df = add_defender_tracking_features(test_df)


def add_engineered_features(df):
    df = df.with_columns(
        (
            pl.col("def_dist_1s") - pl.col("def_dist_025s")
        ).alias("defender_close_1s")
    )

    df = df.with_columns(
        (
            pl.col("def_dist_05s") - pl.col("def_dist_025s")
        ).alias("defender_close_05s")
    )

    df = df.with_columns(
        (
            (pl.col("def_dist_075s") - pl.col("def_dist_025s")) / 0.5
        ).alias("defender_close_rate")
    )

    df = df.with_columns(
        (pl.col("distance") ** 2).alias("distance_squared")
    )

    df = df.with_columns(
        (
            pl.col("distance") * pl.col("three").cast(pl.Int8)
        ).alias("three_distance")
    )

    return df


train_df = add_engineered_features(train_df)
test_df = add_engineered_features(test_df)


numeric_features = [
    "distance",
    "distance_squared",
    "locationx",
    "locationy",
    "shotclock",
    "dribblesbefore",
    "closestdefdist",
    "shooterspeed",
    "num_contesters",
    "def_dist_1s",
    "def_dist_075s",
    "def_dist_05s",
    "def_dist_025s",
    "defender_close_1s",
    "defender_close_05s",
    "defender_close_rate",
    "three_distance"
]


categorical_features = [
    "season_id",
    "off_team_id",
    "def_team_id",
    "shooter_id",
    "closestdef_id",
    "month",
    "gamestate",
    "shottype",
    "contested",
    "three"
]


features = numeric_features + categorical_features


X = train_df.select(features).to_pandas()
y = train_df["outcome"].to_pandas()
X_test = test_df.select(features).to_pandas()


for column in categorical_features:
    X[column] = X[column].astype(str)
    X_test[column] = X_test[column].astype(str)


categorical_indices = [
    X.columns.get_loc(column)
    for column in categorical_features
]


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Logloss",
    iterations=400,
    learning_rate=0.07,
    depth=7,
    l2_leaf_reg=5,
    random_seed=42,
    thread_count=-1,
    early_stopping_rounds=50,
    verbose=50
)


print("\nTraining CatBoost model...\n")


model.fit(
    X_train,
    y_train,
    cat_features=categorical_indices,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)


valid_probs = model.predict_proba(X_valid)[:, 1]

validation_logloss = log_loss(
    y_valid,
    valid_probs
)


print("\n========================================")
print("Validation log-loss:", validation_logloss)
print("Best iteration:", model.get_best_iteration())
print("========================================")


importance = model.get_feature_importance()


importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
})


importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


print("\nTop features:")

print(
    importance_df
    .head(20)
    .to_string(index=False)
)


best_iteration = model.get_best_iteration()


if best_iteration is None or best_iteration < 1:
    best_iteration = 400


final_model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=best_iteration + 1,
    learning_rate=0.07,
    depth=7,
    l2_leaf_reg=5,
    random_seed=42,
    thread_count=-1,
    verbose=50
)


print("\nTraining final model on all training data...\n")


final_model.fit(
    X,
    y,
    cat_features=categorical_indices
)


test_probs = final_model.predict_proba(X_test)[:, 1]


test_probs = test_probs.clip(
    0.000001,
    0.999999
)


submission = pd.DataFrame({
    "shot_id": test_df["shot_id"].to_list(),
    "make_prob": test_probs
})


submission_path = project_folder / "submission.csv"


submission.to_csv(
    submission_path,
    index=False
)