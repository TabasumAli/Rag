import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Initialize Groq API client with your API key
secrets  = st.secrets["groq_api_key"]
client = Groq(api_key=secrets)

# Load the dataset
dataset_path = 'dataseter.csv'  # Ensure the correct path
data = pd.read_csv(dataset_path)

# Preprocess the data (convert categorical variables and all columns to string)
def preprocess_data(df):
    return df.astype(str)

data = preprocess_data(data)

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the dataset (combining features into a single string to represent each row)
data_embeddings = model.encode(data.drop(columns='LUNG_CANCER').agg(' '.join, axis=1).tolist())

# Function to retrieve the most relevant row based on user input
def retrieve_data(user_input):
    user_input_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_input_embedding, data_embeddings)
    most_similar_idx = similarities.argmax()
    return data.iloc[most_similar_idx]

# Function to send data to Groq and get prediction
def get_prediction_from_groq(data_row):
    # Format the input data into a natural language prompt for Groq
    prompt = (
        "Based on the following data about a patient, determine if they are likely to have lung cancer:\n\n"
        f"GENDER: {data_row['GENDER']}\n"
        f"AGE: {data_row['AGE']}\n"
        f"SMOKING: {data_row['SMOKING']}\n"
        f"YELLOW FINGERS: {data_row['YELLOW_FINGERS']}\n"
        f"ANXIETY: {data_row['ANXIETY']}\n"
        f"PEER PRESSURE: {data_row['PEER_PRESSURE']}\n"
        f"CHRONIC DISEASE: {data_row['CHRONIC_DISEASE']}\n"
        f"FATIGUE: {data_row['FATIGUE']}\n"
        f"ALLERGY: {data_row['ALLERGY']}\n"
        f"WHEEZING: {data_row['WHEEZING']}\n"
        f"ALCOHOL CONSUMING: {data_row['ALCOHOL_CONSUMING']}\n"
        f"COUGHING: {data_row['COUGHING']}\n"
        f"SHORTNESS OF BREATH: {data_row['SHORTNESS_OF_BREATH']}\n"
        f"SWALLOWING DIFFICULTY: {data_row['SWALLOWING_DIFFICULTY']}\n"
        f"CHEST PAIN: {data_row['CHEST_PAIN']}\n\n"
        "Is the patient likely to have lung cancer? Respond with 'Yes' or 'No'."
    )

    # Send the prompt to Groq for processing
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",  # Use the appropriate model ID
    )

    # Extract prediction result from response
    return chat_completion.choices[0].message.content.strip()

# Streamlit UI
st.title("Lung Cancer Prediction Using RAG Model")

# User inputs
age = st.slider("Age", 18, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
anxiety = st.selectbox("Anxiety", ["Yes", "No"])
peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])
allergy = st.selectbox("Allergy", ["Yes", "No"])
wheezing = st.selectbox("Wheezing", ["Yes", "No"])
alcohol_consuming = st.selectbox("Alcohol Consuming", ["Yes", "No"])
coughing = st.selectbox("Coughing", ["Yes", "No"])
shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

# Combine inputs into a single string (to use for retrieval)
user_input = f"{gender} {age} {smoking} {yellow_fingers} {anxiety} {peer_pressure} {chronic_disease} {fatigue} {allergy} {wheezing} {alcohol_consuming} {coughing} {shortness_of_breath} {swallowing_difficulty} {chest_pain}"

# Button to submit and process prediction
if st.button("Submit"):
    # Step 4: Retrieve the most relevant data row based on user input
    relevant_data = retrieve_data(user_input)

    # Display the relevant data (for user reference)
    st.write("Most Relevant Data Retrieved from Dataset:")
    st.write(relevant_data)

    # Step 5: Get prediction based on the retrieved data
    prediction = get_prediction_from_groq(relevant_data)

    # Show the prediction result
    st.write("Prediction Result: ", prediction)
