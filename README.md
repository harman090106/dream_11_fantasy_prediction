🏏 Dream11 Fantasy Cricket Prediction
Live Demo 👉 nimuni.streamlit.app

This project predicts fantasy points for individual cricket players using machine learning models trained on past match data. You can select two teams and a stadium, and the app recommends the optimal 11 players based on data-driven predictions.

🚀 Features
Predicts Dream11 points for each player using their historical data.

Accounts for:

Player's recent form.

Opponent matchup.

Venue (stadium) impact.

Generates optimized best 11 using performance scores.

Trained individual regression models for each player using RandomForestRegressor.

Supports live use through a Streamlit web app.

📊 Tech Stack
Category	Tools
Language	Python
ML Models	Scikit-learn (RandomForestRegressor)
Preprocessing	Pandas, NumPy, Scikit-learn
Model Tuning	GridSearchCV
Deployment	Streamlit
Persistence	Joblib (for saving models)

📁 Folder Structure
bash
Copy
Edit
.
models/                   # Folder storing individual player models (.pkl)
  data/                    # Raw and preprocessed data files
  app.py                   # Main Streamlit app
  utils.py                 # Helper functions for preprocessing and modeling
  requirements.txt         # All dependencies
  README.md                # This file

🧠 How It Works
Data Preprocessing:
Dataset contains: Player, Team, Opponent, Stadium, Matches, Runs, Wickets, etc.

Categorical encoding using OneHotEncoder for Stadium, and Team vs Opponent.

Feature scaling with StandardScaler.

Model Training:
Trains a separate Random Forest regression model for each player.

Performs GridSearchCV for hyperparameter tuning (n_estimators, max_depth).

Ignores players with fewer than 5 matches.

Prediction:
For a given team matchup and stadium, predict points for all players.

Select top 11 players with maximum predicted Dream11 points.

📌 Example Inputs
Feature	Example
Team	India
Opponent	Australia
Stadium	Wankhede
Matches Played	12
Batting Avg	42.5
Bowling Econ	6.1

📈 Sample Output
plaintext
Copy
Edit
✅ Optimized Best 11:
1. V. Kohli –  57.3 points
2. R. Sharma – 48.6 points
3. J. Bumrah – 42.0 points
...
Also downloadable as CSV!

🛠️ Setup Instructions
bash
Copy
Edit
git clone https://github.com/yourusername/dream11-fantasy-predictor.git
cd dream11-fantasy-predictor
pip install -r requirements.txt
streamlit run app.py
📌 To-Do
 Add ensemble blending models (XGBoost + RF + SVR).

 Add injury reports & toss impact.

 UI improvements.

 Integrate live match API.

🙋‍♂️ Author
Harman, B.Tech CSE, NIT Kurukshetra

🏁 License
This project is open-source under the MIT License.








Ask ChatGPT
