from data_loader import load_data, encode_features
from model_trainer import train_models
from predictor import load_models, predict_best_11
import os
teams = {'Australia': ['Aaron Hardie','Adam Zampa','Alex Carey','Ben Dwarshuis','Cooper Connolly','Glenn Maxwell','Jake Fraser-McGurk','Josh Inglis','Marnus Labuschagne','Matthew Short','Nathan Ellis','Sean Abbott','Spencer Johnson','Steven Smith','Tanveer Sangha','Travis Head'],
 'Pakistan': ['Abrar Ahmed','Agha Salman','Babar Azam','Faheem Ashraf','Fakhar Zaman','Haris Rauf','Imam ul-Haq','Kamran Ghulam','Khushdil Shah','Mohammad Hasnain','Mohammad Rizwan','Naseem Shah','Saud Shakeel','Shaheen Afridi','Tayyab Tahir','Usman Khan'],
 'England': ['Adil Rashid','Ben Duckett','Brydon Carse','Gus Atkinson','Harry Brook','Jamie Overton','Jamie Smith','Joe Root','Jofra Archer','Jos Buttler','Liam Livingstone','Mark Wood','Philip Salt','Rehan Ahmed','Saqib Mahmood','Tom Banton'],
  'South Africa': ['Aiden Markram','Corbin Bosch','David Miller','Heinrich Klaasen','Kagiso Rabada','Keshav Maharaj','Lungi Ngidi','Marco Jansen','Rassie van der-Dussen','Ryan Rickelton','Tabraiz Shamsi','Temba Bavuma','Tony de Zorzi','Tristan Stubbs','Wiaan Mulder'],
  'India': ['Arshdeep Singh','Axar Patel','Hardik Pandya','Harshit Rana','Kuldeep Yadav','Lokesh Rahul','Mohammed Shami','Ravindra Jadeja','Rishabh Pant','Rohit Sharma','Shreyas Iyer','Shubman Gill','Varun Chakravarthy','Virat Kohli','Washington Sundar'],
 'Afghanistan': ['Azmatullah Omarzai','Bilal Sami','Darwish Rasooli','Fareed Ahmed Malik','FazalHaq Farooqi','Gulbadin Naib','Hashmatullah Shahidi','Ibrahim Zadran','Ikram Alikhil','Mohammad Nabi','Nangeyalia Kharote','Naveed Zadran','Noor Ahmad','Rahmanullah Gurbaz','Rahmat Shah','Rashid-Khan','Sediqullah Atal'],
 'New Zealand': ['Daryl Mitchell','Devon Conway','Glenn Phillips','Jacob Duffy','Kane Williamson','Kyle Jamieson','Mark Chapman','Matt Henry','Michael Bracewell','Mitchell Santner','Nathan Smith','Rachin Ravindra','Tom Latham','Will Young',"William O'Rourke"],
 'Bangladesh': ['Jaker Ali Anik','Mehidy Hasan Miraz','Mohammad Mahmudullah','Mohammad Parvez Hossain-Emon','Mushfiqur Rahim','Mustafizur Rahman','Nahid Rana','Najmul Hossain Shanto','Nasum Ahmed','Rishad- Hossain','Soumya Sarkar','Tanzid Hasan','Tanzim Sakib','Taskin Ahmed','Tawhid Hridoy']}


# Define paths
DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"

# ----- Load Data -----
print("Loading data.....")
df = load_data(DATA_FOLDER)
df, encoder = encode_features(df)
print("Data loading completed")

# ----- Train & Save Models -----
player_models = train_models(df)

# ----- Train or Load Models -----
if not os.path.exists(MODEL_FOLDER) or len(os.listdir(MODEL_FOLDER)) == 0:
    print("🛠️ No models found! Training new models...")
    player_models = train_models(df, save_path=MODEL_FOLDER)
else:
    print("✅ Found pre-trained models. Loading them...")
    player_models = load_models(MODEL_FOLDER)

# ----- Predict Best XI -----
models = load_models(MODEL_FOLDER)
stadium = "Dubai (DISC)"
team1 = "India"
team2 = "New Zealand"
match_squad = teams[team1]+teams[team2]

best_xi = predict_best_11(stadium, team1, team2, match_squad, df, encoder, models)

print("\n🏏 Best 11 Players for the Match:")
for player, points in best_xi:
    print(f"{player}: {points:.2f} points")

