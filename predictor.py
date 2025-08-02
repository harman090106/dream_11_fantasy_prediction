import pandas as pd
import joblib
import os
from player_to_country import player_to_country
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

team_composition={
        "Dubai (DISC)":{
            "Batsman" :  4,
            "Spinner" : 1,
            "Wicket Keeper":1,
            "Spin Rounder": 2,
            "Fast Rounder":1,
            "Faster": 2
        },
        "Lahore":{
            "Batsman" :  5,
            "Spinner" : 1,
            "Wicket Keeper":1,
            "Spin Rounder": 1,
            "Fast Rounder":1,
            "Faster": 3
        },
        "Karachi":{
            "Batsman" :  4,
            "Spinner" : 2,
            "Wicket Keeper":1,
            "Spin Rounder": 1,
            "Fast Rounder":1,
            "Faster": 2
        },
        "Rawalpindi":{
            "Batsman" :  4,
            "Spinner" : 1,
            "Wicket Keeper":1,
            "Spin Rounder": 1,
            "Fast Rounder":1,
            "Faster": 3
        }   
    }




def load_models(model_path="models/"):
    """Loads all trained models from disk."""
    models = {}
    for file in os.listdir(model_path):
        if file.endswith(".pkl"):
            player_name = file.replace(".pkl", "")
            models[player_name] = joblib.load(os.path.join(model_path, file))
    return models




def predict_best_11(stadium, team1, team2, squad, df, encoder, player_models):
    """
    Predicts the best 11 players for a given match, ensuring that:
    - Career and recent stats are correctly included.
    - Missing recent averages are handled properly.
    - The correct opponent is mapped.
    """
    predictions = []

    for player in squad:
        if player not in player_models:
            continue  # Skip if the player's model is missing

        model = player_models[player]

        # Find player's team
        player_team = player_to_country.get(player, "Unknown")

        # Determine opponent
        opponent = team1 if player_team == team2 else team2
        team_opponent = player_team + "_vs_" + opponent

        # Encode stadium & team-opponent
        input_data = pd.DataFrame([[stadium, team_opponent]], columns=["Stadium", "Team_Opponent"])
        encoded_input = encoder.transform(input_data)
        encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(["Stadium", "Team_Opponent"]))

        # Get latest recent stats & fill missing values with career stats
        player_data = df[df["Player"] == player]

        if player_data.empty:
            continue  # Skip if no data found for player

        # Get most recent available values
        final_avg_batting = player_data["Recent Batting Avg"].dropna().values[-1] if not player_data["Recent Batting Avg"].dropna().empty else player_data["Career Batting Avg"].dropna().values[-1] if not player_data["Career Batting Avg"].dropna().empty else 0
        final_wickets = player_data["Recent Wickets"].dropna().values[-1] if not player_data["Recent Wickets"].dropna().empty else player_data["Career Wickets"].dropna().values[-1] if not player_data["Career Wickets"].dropna().empty else 0

        # Combine all features
        input_features = pd.DataFrame([[final_avg_batting, final_wickets]], 
                                      columns=["Recent Batting Avg", "Recent Wickets"])
        input_features = pd.concat([input_features, encoded_df], axis=1)

        # Ensure columns match the trained model
        missing_cols = set(df.columns) - set(input_features.columns)
        for col in missing_cols:
            input_features[col] = 0  # Add missing columns with default 0

        input_features = input_features[df.columns.drop(["Dream11 Points", "Player"])]  # Ensure order

        # Predict Dream11 Points
        predicted_points = model.predict(input_features)[0]
        predictions.append((player, predicted_points))

    # Sort players by predicted points & select top 11
    best_11 = sorted(predictions, key=lambda x: x[1], reverse=True)
    return get_optimised_11(best_11,team1,team2,pd.read_excel("champ_players.xlsx"),stadium)


def get_optimised_11(sorted_11,country1,country2,players_name,stadium):
    players=[]
    composition=team_composition[stadium]
    for player,score in sorted_11:
        player_row = players_name[players_name.player == player]

        players.append({
            "name":player,
            "role":player_row['Role'].values[0] if not player_row.empty else "Unknown",
            "fantasy_points":score,
            "credits":player_row['Credits'].values[0],
            "team":player_row['Country']
        })
    # print(players[1],"----------------")
    problem = LpProblem("Fantasy_Team_Selection", LpMaximize)
    
    player_vars = {p["name"]: LpVariable(p["name"], cat="Binary") for p in players}
    
    problem += lpSum(player_vars[p["name"]] * p["fantasy_points"] for p in players)
    
    problem += lpSum(player_vars[p["name"]] for p in players) == 11
    
    problem += lpSum(player_vars[p["name"]] * p["credits"] for p in players) <= 120
    
    problem += lpSum(player_vars[p["name"]] for p in players if p["role"] == "Batsman") >= composition["Batsman"]
    
    problem += lpSum(player_vars[p["name"]] for p in players if p["role"] == "Wicket Keeper") >= composition["Wicket Keeper"]
    
    problem += lpSum(player_vars[p["name"]] for p in players if p["role"] == "Spin Rounder") == composition["Spin Rounder"]
    problem += lpSum(player_vars[p["name"]] for p in players if p["role"] == "Fast Rounder") == composition["Fast Rounder"]
    
    problem += lpSum(player_vars[p["name"]] for p in players if p["role"] == "Faster") <= composition["Faster"]
    problem += lpSum(player_vars[p["name"]] for p in players if p["role"] == "Spinner") <= composition["Spinner"]
    
    problem += lpSum(player_vars[p["name"]] for p in players if p["team"].values[0] == country1) >= 2
    problem += lpSum(player_vars[p["name"]] for p in players if p["team"].values[0] == country2) >= 2

    
    # Solve the LP problem
    problem.solve()
    
    # Get the selected players
    selected_team = [(p["name"],p["role"],p["fantasy_points"]) for p in players if player_vars[p["name"]].value() == 1]
    
    return selected_team