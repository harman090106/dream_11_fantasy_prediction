# 🏏 Dream11 Fantasy Prediction Project

This project predicts the best possible playing 11 for a Dream11 fantasy cricket match using historical data, player stats, recent form, and venue-based performance.

## 🔍 Problem Statement

Fantasy cricket requires users to select players based on skill, form, and conditions. However, manually analyzing all variables is time-consuming and error-prone.  
This project automates the process using machine learning and statistical modeling to generate an optimal Dream11 team for any given match.

## 🚀 Features

- Predicts the top 11 players for an upcoming match
- Uses historical player performance data
- Considers recent form, venue, match type (ODI/T20), and opposition
- Optimizes team composition based on Dream11 constraints (credits, roles, etc.)
- Built with a modular ML pipeline

## 🧠 Tech Stack

- **Language:** Python  
- **ML Libraries:** scikit-learn, pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Others:** Jupyter Notebooks, MongoDB (for storing player data)

## 📁 Folder Structure

dream_11/
├── data/ # Raw + processed datasets
├── notebooks/ # EDA, model training, evaluation
├── src/ # Source code for training, prediction
├── utils/ # Helper functions
├── saved_models/ # Trained model files
├── requirements.txt
└── README.md

markdown
Copy
Edit

## 🧪 How to Run

1. Clone the repo:
git clone https://github.com/your-username/dream_11_fantasy_prediction.git

markdown
Copy
Edit

2. Install requirements:
pip install -r requirements.txt

markdown
Copy
Edit

3. Run the prediction script:
python src/predict_best_11.py

pgsql
Copy
Edit

## 📊 Sample Output

- Predicted best 11 players with roles and scores
- Justification based on recent form, venue stats, etc.
- Credit optimization logic

## 📌 Future Improvements

- UI for user inputs (teams, venue)
- Real-time data scraping from cricket APIs
- Integration with fantasy platforms for automated team upload

## 🤝 Contribution

Open to collaborations and suggestions. Feel free to fork or raise issues!

## 📜 License

MIT License