import pandas as pd
import glob
import os
from sklearn.preprocessing import OneHotEncoder
from player_to_country import player_to_country  # Import player-team mapping

def load_data(data_path):
    """Loads player stats from Excel files and processes them."""
    file_paths = glob.glob(os.path.join(data_path, "*.xlsx"))
    all_data = []

    for file_path in file_paths:
        player_name = os.path.basename(file_path).replace(".xlsx", "")
        sheet2_df = pd.read_excel(file_path, sheet_name="Sheet2")

        # Add Player & Team columns
        sheet2_df["Player"] = player_name
        sheet2_df["Team"] = sheet2_df["Player"].map(player_to_country)
        all_data.append(sheet2_df)

    df = pd.concat(all_data, ignore_index=True)
    df.columns = ["Opponent", "Stadium", "Recent Batting Avg", "Career Batting Avg", 
                  "Recent Wickets", "Career Wickets", "Dream11 Points", "Player", "Team"]
    
    df.fillna(0, inplace=True)  # Handle missing values
    return df

def encode_features(df):
    """Encodes categorical features (Stadium, Team vs Opponent)."""
    df["Team_Opponent"] = df["Team"].astype(str) + "_vs_" + df["Opponent"].astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_features = df[["Stadium", "Team_Opponent"]]
    cat_encoded = encoder.fit_transform(cat_features)

    encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(["Stadium", "Team_Opponent"]))

    df = df.drop(["Stadium", "Opponent", "Team_Opponent",'Team'], axis=1).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder
