import pickle



import streamlit as st
import pandas as pd


teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open("pipe.pkl","rb"))
st.title("IPL Win Predictor")

col1, col2 = st.columns(2)

with col1:
    Batting_Team = st.selectbox("Select the Batting Team",sorted(teams))
with col2:
    Bowling_Team = st.selectbox("Select the Bowling Team",sorted(teams))

selected_city = st.selectbox("Select Host City",sorted(cities))

target = st.number_input("Target")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input("Score")
with col4:
    overs = st.number_input("Overs Completed")
with col5:
    wickets = st.number_input("Wicket Out")

if st.button("Predict Probability"):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({
        "batting_team": [Batting_Team],
        "bowling_team": [Bowling_Team],
        "city": [selected_city],
        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets": [wickets],  # Corrected here
        "total_runs_x": [target],
        "crr": [crr],
        "rrr": [rrr]
    })

    st.table(input_df)

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.header(Batting_Team + "- " + str(round(win*100)) + "%")
    st.header(Bowling_Team + "- " + str(round(loss*100)) + "%")