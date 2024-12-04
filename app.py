import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableLambda

# Set API key for Google Generative AI
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDqCKy6Kfhaz1nnJ891e6MvqZXGrwkv6Xw'

generation_config = {"temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)


prompt_template_diet = PromptTemplate(
    input_variables=['name', 'age', 'gender', 'weight', 'height', 'diet_type', 'health_condition', 'region', 'allergies'],
    template="Based on the following details:\n"
             "Name: {name}\n"
             "Age: {age}\n"
             "Gender: {gender}\n"
             "Weight: {weight} kg\n"
             "Height: {height} cm\n"
             "Diet Type (Veg/Non-Veg): {diet_type}\n"
             "Health Condition: {health_condition}\n"
             "Region: {region}\n"
             "Allergies: {allergies}\n"
             "Provide:\n"
             "- 3 personalized meal recommendations (breakfast, lunch, dinner)\n"
             "- 3 snacks\n"
             "- 3 fitness tips tailored to the individual."
)


chain_diet = RunnableLambda(lambda inputs: prompt_template_diet.format(**inputs)) | model

st.markdown(
    """
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
            text-align: center;
            color: #FFFFFF;
            margin-bottom: 20px;
            background: #4CAF50;
            padding: 10px;
            border-radius: 10px;
        }
        .subtitle {
            font-size: 20px;
            font-family: 'Helvetica', sans-serif;
            text-align: center;
            color: #FFFFFF;
            margin-bottom: 30px;
            background: #2196F3;
            padding: 10px;
            border-radius: 10px;
        }
        .form-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
        }
        .recommendations {
            font-family: 'Helvetica', sans-serif;
            margin-top: 20px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            transition: transform 0.3s ease;
        }
        .stButton button:hover {
            transform: scale(1.05);
            background-color: #45a049;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Personalized Diet Recommendations</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get tailored meal plans and fitness tips</div>', unsafe_allow_html=True)

#input
with st.form(key='diet_input_form', clear_on_submit=True):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    name = st.text_input('Name:', placeholder='Enter your name')
    age = st.number_input('Age:', min_value=1, max_value=120, step=1)
    gender = st.selectbox('Gender:', ['Male', 'Female', 'Other'])
    weight = st.number_input('Weight (kg):', min_value=1, step=1)
    height = st.number_input('Height (cm):', min_value=1, step=1)
    diet_type = st.selectbox('Diet Type:', ['Veg', 'Non-Veg'])
    health_condition = st.text_input('Health Condition:', placeholder='Mention any existing health issues')
    region = st.text_input('Region:', placeholder='Enter your region')
    allergies = st.text_input('Allergies:', placeholder='List any known allergies')

    submit_button = st.form_submit_button(label='Get Recommendations')
    st.markdown('</div>', unsafe_allow_html=True)

# Handling submission
if submit_button:
    # necessery to fill all requirements
    if all([name, age, gender, weight, height, diet_type, health_condition, region, allergies]):
        input_data = {
            'name': name,
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'diet_type': diet_type,
            'health_condition': health_condition,
            'region': region,
            'allergies': allergies
        }
    
        recommendations = chain_diet.invoke(input_data)

        # Display 
        st.markdown('<div class="subtitle">Your Recommendations:</div>', unsafe_allow_html=True)
        st.markdown('<div class="recommendations">', unsafe_allow_html=True)
        st.markdown(recommendations, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Please fill in all fields to get recommendations.")
