from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tensorflow as tf
import numpy as np
import os
import google.generativeai as genai
import base64
import time

# Path to your local image
image_path = "./web_app_imgs/background.jpg"

# Encode the image to base64
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Create the CSS with the base64 encoded image
page_element = f"""
<style>
[data-testid="stAppViewContainer"]{{
  background-image: url("data:image/jpeg;base64,{encoded_string}");
  background-size: cover;
}}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(page_element, unsafe_allow_html=True)


# tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('./trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # to convert single image to a batch
    prediction = model.predict([input_arr])
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Us", "Disease Recognition", "Chat With Us!"])

# Home Page
if (app_mode=="Home"):
    st.header("IndiGro: Predict. Prevent. Prosper.")
    image_path = "./web_app_imgs/home1.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    
    ### 🪴Our Mission
    At IndiGro, our mission is to revolutionize crop protection by empowering farmers with cutting-edge AI-powered disease diagnosis. We aim to provide farmers with a fast, reliable, and cost-effective solution for identifying plant diseases in their fields, enabling them to take timely action to minimize crop losses and maximize yields. By harnessing the power of technology, we strive to contribute to food security, sustainable farming practices, and the overall profitability of agricultural operations.
    
    ### 🏹How It Works?
    1. **Upload Images:** Simply upload the images to our platform.
    2. **AI Analysis:** Our AI model, trained on extensive data, analyzes the images to identify the presence of diseases in crops.
    3. **Rapid Diagnosis:** Our AI-powered system provides instant disease diagnosis, allowing you to act swiftly to protect your crops.
    4. **Take Action:** Armed with this information, farmers can take swift and targeted action to control the spread of disease and optimize their harvest.
    
    ### ✨Why Choose Us?
    1. **Swift Diagnosis:** Our AI-powered system swiftly identifies diseases, enabling prompt action to safeguard your crops.
    2. **Unrivaled Accuracy:** Backed by extensive datasets, our model ensures precise detection of plant diseases.
    3. **Cost-effective Solution:** Save valuable time and resources with our efficient approach compared to traditional lab testing.
    4. **Empowering Farmers:** We equip farmers with advanced technology, empowering them to make informed decisions and enhance their farm management practices.
    
    ### 🪴Get Started 
    At IndiGro, we are a team of passionate students dedicated to applying the latest advancements in AI and machine learning to tackle critical issues in agriculture. With a blend of expertise in AI, agriculture, and data science, we are committed to crafting innovative solutions to empower farmers and bolster food security globally. Our relentless pursuit of excellence motivates us to refine and enhance our platform continually, ensuring that farmers have the necessary tools and resources to thrive in today's dynamic agricultural environment. Join us on our journey to revolutionize crop protection and shape the future of farming.
    
    """)
    
# About Page
elif (app_mode=="About Us"):
    st.header("About Us")
    
    image_path = "./web_app_imgs/about.jpg"
    st.image(image_path, use_column_width=True)
    
    st.markdown("""
    
    At IndiGro, our team is comprised of driven and passionate students who are united by a common goal: harnessing the latest advancements in artificial intelligence (AI) and machine learning to tackle the critical issues facing agriculture today. 🌱 With a unique blend of expertise spanning AI, agriculture, and data science, we are deeply committed to developing innovative solutions that not only empower farmers but also contribute to the global efforts aimed at enhancing food security. 🌍

    Our relentless pursuit of excellence serves as the driving force behind our work. We are dedicated to continually refining and enhancing our platform, ensuring that it remains at the forefront of agricultural technology. By providing farmers with access to cutting-edge tools and resources, we aim to equip them with the necessary means to thrive in today's rapidly evolving agricultural landscape. 🚜

    In our pursuit of excellence, we have trained our AI model on an extensive dataset consisting of 35,406 training images, meticulously curated to ensure optimal performance. Additionally, we have rigorously validated our model using 17,572 validation images, further enhancing its accuracy and reliability. With a thorough testing phase that includes 33 images, we are confident in the effectiveness of our solution and its potential to revolutionize crop protection practices worldwide. 📊 Join us as we embark on this transformative journey to revolutionize crop protection and pave the way for a more sustainable and prosperous future in farming. 🌟
    
    """)
    
# Prediction Page
elif (app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image: ")
    if (st.button("Show Image")):
        st.image(test_image, use_column_width=True)
    # Predict Button
    if (st.button("Predict")):
        with st.spinner('Predicting...'):
            time.sleep(3)
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Define Class
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success("The plant appears to have a {} condition.".format(class_name[result_index]))
        
# Chat With Us Page
elif (app_mode=="Chat With Us!"):
        
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # function to load Gemini Pro model and get response
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    
    def get_gemini_response(question):
        response = chat.send_message(question, stream=True)
        return response
    
    # initialize streamlit app
    st.header("Welcome to IndiGro! Let's help your plants thrive!")
    
    # initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
       st.session_state['chat_history'] = []
       
    input = st.text_input("Ask me Something! ", key="input")
    submit = st.button("Click to Ask!")
    
    if submit and input:
        response = get_gemini_response(input)
        # add user query and response to session chat history
        st.session_state['chat_history'].append(("You", input))
        st.subheader("The Response is ")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot", chunk.text))
    st.subheader("Chat History:")
    
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}:{text}")