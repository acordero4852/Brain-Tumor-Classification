import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

import res
import utils as ut

# Set page config for the app
st.set_page_config(page_title="Brain Tumor Classification")

# Add a custom CSS style to the app
st.markdown(
    """
      <style>
        [data-testid="stDecoration"] {
          background-image: linear-gradient(90deg, #52ff76, #00b4d8);
        }
      </style>
    """,
    unsafe_allow_html=True,
)

# Add a title and description to the app
st.title("Brain Tumor Classification")

st.write("Upload an image of a brain MRI scan to classify.")

# Add a file uploader to the app
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if a file has been uploaded
if uploaded_file is not None:

    # Select the model to use for classification
    selected_model = st.radio(
        "Select a model", ("Transfer Learning - Xception", "Custom CNN")
    )

    # Load the selected model and set the image size
    if selected_model == "Transfer Learning - Xception":
        model = res.load_xception_model("./models/xception_model.weights.h5")
        img_size = (299, 299)
    else:
        model = load_model("./models/cnn_model.h5")
        img_size = (224, 224)

    # Load the image and preprocess it
    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make a prediction using the model
    prediction = model.predict(img_array)

    # Get the class index and result
    class_index = np.argmax(prediction[0])
    result = labels[class_index]

    # Generate the saliency map
    saliency_map = res.generate_saliency_map(
        model, img, uploaded_file, img_array, class_index, img_size
    )

    # Display the uploaded image and the saliency map
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(saliency_map, caption="Saliency Map", use_column_width=True)

    # Display the classification results
    st.write("## Classification Results")

    ut.create_result_container(result, prediction, class_index)

    # Display the probabilities chart
    probabilties = prediction[0]
    sorted_indices = np.argsort(probabilties)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probabilities = probabilties[sorted_indices]

    fig = ut.create_probabilities_chart(result, sorted_probabilities, sorted_labels)

    st.plotly_chart(fig)

    # Generate the explanation
    saliency_map_path = f"saliency_maps/{uploaded_file.name}"
    explanation = res.generate_explanation(
        saliency_map_path, result, prediction[0][class_index]
    )

    st.write("## Explanation")
    st.write(explanation)
