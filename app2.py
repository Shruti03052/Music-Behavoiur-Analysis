
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load your original music listening data for the Streamlit app
df = pd.read_csv('C:\\Users\\Shruti\\Desktop\\cp\\de_lab_5\\Music prediction - Listeners - Sheet1.csv')

# Load the training dataset for genre prediction
train_data = pd.read_csv('C:\\Users\\Shruti\\Desktop\\cp\\de_lab_5\\train.csv')

# Drop irrelevant columns from the training data
train_data = train_data.drop(['instrumentalness', 'key'], axis=1)

# Define selected features for prediction (features)
selected_columns = ['Popularity', 'danceability', 'energy', 'loudness', 
                     'speechiness', 'valence', 'liveness', 
                     'tempo', 'duration_in min/ms','time_signature']

# Prepare the features (X) and target (y) for the Random Forest model
X = train_data[selected_columns]
y = train_data['Class']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define the mapping of numeric class labels to genre names
music_genre = ["Acoustic/Folk", "Alt_Music", "Blues", "Bollywood", "Country", 
               "HipHop", "Indie Alt", "Instrumental", "Metal", "Pop", "Rock"]

# Streamlit app title
st.title("Music Listening Data Analysis")

# Sidebar for navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Select View", ["Summary Statistics", "Group by Mood", "Activity Patterns", "User Frequency", "Top Users", "Genre Prediction"])

# Option for Genre Prediction
if option == "Genre Prediction":
    st.subheader("Predict the Genre of a Music Track")
    
    # Create input fields for the relevant features
    popularity = st.number_input("Popularity", min_value=0, max_value=100, value=50)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness", -40.0, 0.0, -10.0)
   # mode = st.selectbox("Mode", ['Major', 'Minor'])
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
    tempo = st.number_input("Tempo", min_value=50, max_value=200, value=120)
    duration = st.number_input("Duration (ms)", min_value=100000, max_value=500000, value=300000)
    time_signature = st.selectbox("Time Signature", [1, 2, 3, 4])

    # Create a button that will trigger prediction
    if st.button('Predict Genre'):
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'Popularity': [popularity],
            'danceability': [danceability],
            'energy': [energy],
            'loudness': [loudness],
           # 'mode': [mode],
            'speechiness': [speechiness],
            'valence': [valence],
            'liveness': [liveness],
            'tempo': [tempo],
            'duration_in min/ms': [duration],
            'time_signature': [time_signature]
        })

        # Encode the 'mode' column to numeric (Major: 1, Minor: 0)
      #  input_data['mode'] = input_data['mode'].map({'Major': 1, 'Minor': 0})

        # Predict the genre (numerically)
        genre_prediction = model.predict(input_data)

        # Map the numeric prediction to genre names
        predicted_genre = music_genre[genre_prediction[0]]

        # Display the predicted genre
        st.write(f"The predicted genre is: **{predicted_genre}**")

# Option for Summary Statistics
elif option == "Summary Statistics":
    st.subheader("Summary Statistics")
    
    # Calculate the summary statistics
    summary_stats = df.describe() 
    selected_columns = ['Mood', 'SessionLength', 'ListeningFrequency', 'TimeOfDay']
    specific_stats = summary_stats.loc[['count', 'mean', 'std', 'max','min'], selected_columns]

    # Display the specific statistics in Streamlit
    st.write(specific_stats)


# Option for Group by Mood
elif option == "Group by Mood":
    st.subheader("Group by Mood")
    mood_summary = df.groupby('Mood').agg({
        'SessionLength': 'mean',
        'ListeningFrequency': 'mean',
        'UserName': 'count'
    }).rename(columns={'UserName': 'UserCount'}).reset_index()
    
    st.write(mood_summary)
    st.bar_chart(mood_summary.set_index('Mood')['UserCount'])

# Option for Activity Patterns
elif option == "Activity Patterns":
    st.subheader("User Activity Patterns by Time of Day")
    time_of_day_summary = df.groupby('TimeOfDay').agg({
        'SessionLength': 'mean',
        'ListeningFrequency': 'mean'
    }).reset_index()
    
    st.write(time_of_day_summary)
    st.line_chart(time_of_day_summary.set_index('TimeOfDay')['SessionLength'])

# Option for User Frequency
elif option == "User Frequency":
    st.subheader("Listening Frequency Categories")
    
    # Categorize users based on listening frequency
    def categorize_frequency(freq):
        if freq <= 5:
            return 'Low'
        elif freq <= 15:
            return 'Medium'
        else:
            return 'High'

    df['ListeningCategory'] = df['ListeningFrequency'].apply(categorize_frequency)
    frequency_summary = df.groupby('ListeningCategory').agg({
        'SessionLength': 'mean',
        'UserName': 'count'
    }).rename(columns={'UserName': 'UserCount'}).reset_index()
    
    st.write(frequency_summary)
    st.bar_chart(frequency_summary.set_index('ListeningCategory')['UserCount'])

# Option for Top Users
elif option == "Top Users":
    st.subheader("Top Users by Listening Frequency")
    top_users = df.nlargest(10, 'ListeningFrequency')
    st.write(top_users[['UserName', 'ListeningFrequency']])
