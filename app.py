
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

# Load the model (will be in the same folder when deployed)
@st.cache_resource
def load_model():
    return keras.models.load_model("best_strong(1).h5")

model = load_model()
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️")
st.title("🗑️ Smart Garbage Classifier")
st.write("Custom lightweight CNN trained from scratch | ~74% accuracy on test set")

uploaded_file = st.file_uploader("Upload an image of garbage", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)[0]
    
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # Recyclable check
    recyclable_classes = {"cardboard", "glass", "metal", "paper", "plastic"}
    if predicted_class in recyclable_classes:
        st.success(f"**{predicted_class.upper()}** ✅ This is recyclable! ♻️😄")
    else:
        st.warning(f"**{predicted_class.upper()}** ⚠️ General trash – dispose carefully 🗑️😐")
    
    st.write(f"**Confidence:** {confidence:.1%}")
    
    # Show probability bar chart
    prob_dict = {name: float(pred) for name, pred in zip(class_names, predictions)}
    st.bar_chart(prob_dict)
