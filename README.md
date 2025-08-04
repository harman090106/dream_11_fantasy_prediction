# 🏏 Dream11 Fantasy Prediction Model

Predict the fantasy performance of players in upcoming cricket matches using historical stats, machine learning, and match conditions.

---

**🔗 Live App:** [nimuni.streamlit.app](https://nimuni.streamlit.app)  
**📁 Tech Stack:** Python, Pandas, Scikit-learn, Streamlit, Joblib, GridSearchCV

---

## 🚀 Project Overview

This project predicts Dream11 fantasy points for cricket players using:

- **Past performance**
- **Match conditions**
- **Opponent team**
- **Stadium history**

It helps select an optimized Best XI for fantasy leagues like Dream11, based on predicted scores.

---

## 📊 Data Overview

The model uses two types of data:

- **Match-wise Player Stats**
  - Runs, wickets, boundaries, strike rate, etc.
  - Team and opponent
  - Stadium

- **Target Label**
  - Dream11 Points: Manually calculated based on standard scoring rules.

---

## 🔍 Key Features Used

| Feature                  | Description                                |
|--------------------------|--------------------------------------------|
| Runs, Wickets, Catches   | Player performance stats                   |
| Stadium                  | Location of match                          |
| Team vs Opponent         | Encoded as "Team_vs_Opponent"              |
| Encoded Features         | One-hot encoded team, opponent, and stadium|

_All features are scaled using `StandardScaler`._

---

## 🧠 Model Training

For each player:

1. Filter out their match data.
2. Train a separate `RandomForestRegressor`.
3. Use `GridSearchCV` to find optimal hyperparameters.
4. Skip players with very little match data (e.g., <5 games).
5. Store trained models using `joblib`.

**Sample Code Snippet:**
```python
for player in df["Player"].unique():
    player_df = df[df["Player"] == player]
    # ... preprocessing ...
    model = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
    model.fit(X_train, y_train)
    joblib.dump(model, f"{save_path}/{player}.pkl")

Evaluation

Model accuracy is measured using .score() (R² on test data).
Most good players (with decent past data) show reasonable predictions.
Outliers or debutants tend to have unstable scores (as expected).
🎯 Output
Each player gets a predicted fantasy score.
The app selects an optimized XI using constraints like:
Roles (batsman, bowler, etc.)
Budget
Recent form

💻 Run Locally

git clone https://github.com/your-username/Dream11-Fantasy-Predictor.git
cd Dream11-Fantasy-Predictor
pip install -r requirements.txt
streamlit run app.py

📦 Dependencies
pandas
scikit-learn
streamlit
joblib
🌐 Live Deployment
The app is deployed on Streamlit Cloud and accessible at:
👉 nimuni.streamlit.app

📚 Future Enhancements
Use ball-by-ball data for better micro-analysis.
Include recent player form based on the last 2 months.
Add support for real-time injury/team updates.
Experiment with deep learning (e.g., LSTM for temporal performance trends).
🙋‍♂️ Author
Harmanjeet Singh
Passionate about machine learning, data science, and real-world sports applications.


Let me know if you want this file pushed directly to your repository!
