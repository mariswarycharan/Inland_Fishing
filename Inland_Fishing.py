import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Inland_Fishing",page_icon="🦈")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


st.title("Improving Generalization in Inland Fish Species Classification using Data Augmentation")

# Load the saved CNN model
model = tf.keras.models.load_model('fish.h5')

# Define the mapping of class indices to species labels
species_labels = ['Barramundi','Katla', 'Mirgaal', 'Rohu','SilverCarp','Tilapia','Viraal']

# Define the prediction function
def predict(image):
    # Preprocess the image
    try:
        image = image.resize((64, 64))  # Resize the image to the input size expected by the model
        image = np.array(image)  # Convert the PIL image to a numpy array
        image = image / 255.0  # Normalize the pixel values between 0 and 1
        image = np.expand_dims(image, axis=0)  # Add a batch dimension

        # Make predictions using the loaded CNN model
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)
        return predictions, predicted_class
    except Exception as e:
        print(f"Error occurred during image preprocessing: {str(e)}")
        return None, None

image_upload = st.file_uploader("Upload the image")

submit = st.button("SUBMIT")

if submit:
    
    image = Image.open(image_upload)
    st.image(image,use_column_width=True)
    predictions, predicted_class = predict(image)
    if predictions is not None and predicted_class is not None:
        # Retrieve the predicted species label
        predicted_species = species_labels[predicted_class[0]]

        st.title("Result :")
        
        st.success(predicted_species)
        
        c1,c2 = st.columns(2)
        
        with c1 : st.subheader("species_labels")
        with c2 : st.subheader("Confidence  level")
        
        for i,j in zip(species_labels,predictions[0]):
            
            with c1 : st.info(i)
            with c2 : st.code(round(j*100,4))
      
      

i1,i2 = st.columns(2)          
  

  
# with i1:            
#     st.subheader("Before augmentation")
st.image(Image.open(r"images\WhatsApp Image 2023-06-09 at 5.45.21 PM.jpeg"),use_column_width=True)
#     st.image(Image.open(r"images\WhatsApp Image 2023-06-09 at 5.45.54 PM.jpeg"),use_column_width=True)
    
# with i2:
#     st.subheader("After augmentation")
#     st.image(Image.open(r"images\WhatsApp Image 2023-06-09 at 5.44.08 PM.jpeg"),use_column_width=True)
#     st.image(Image.open(r"images\WhatsApp Image 2023-06-09 at 5.44.38 PM.jpeg"),use_column_width=True)
    
    

