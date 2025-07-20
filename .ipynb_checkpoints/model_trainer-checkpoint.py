import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_models(df, save_path="models/"):
    """Trains individual models for each player and saves them."""
    os.makedirs(save_path, exist_ok=True)
    player_models = {}

    for player in df["Player"].unique():
        player_df = df[df["Player"] == player].drop("Player", axis=1)

        X = player_df.drop(columns=["Dream11 Points"])
        y = player_df["Dream11 Points"]

        if len(player_df) < 5:  # Ignore players with very few matches
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        model_path = os.path.join(save_path, f"{player}.pkl")
        joblib.dump(model, model_path)  # Save model
        player_models[player] = model_path  # Store model path

    print(f"✅ Models trained and saved for {len(player_models)} players.")
    return player_models
