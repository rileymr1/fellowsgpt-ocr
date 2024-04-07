import streamlit as st
import os
import base64
import io

from build_chain import chain_multimodal_rag
from build_chain import retriever
from build_chain import split_image_text_types
from PIL import Image
import numpy as np

from get_external_ip import get_external_ip

from dotenv import load_dotenv
os.environ.clear()
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] # st.sidebar.text_input('OpenAI API Key', type='password')

st.title('🏥 FellowsGPT')

## For use if need to whitelist certain IP addresses
# external_ip = get_external_ip()
# st.write("External IP: ", external_ip)

def base64_to_image(base64_string):
    # Decode base64 string to binary data
    binary_data = base64.b64decode(base64_string)

    # Create an image object from binary data
    image = Image.open(io.BytesIO(binary_data))

    # Convert image to numpy array
    image_array = np.array(image)

    # Check image shape to determine type
    if len(image_array.shape) == 2:
        # Monochrome image
        return image_array
    elif len(image_array.shape) == 3:
        if image_array.shape[2] == 1:
            # Monochrome image (with alpha channel)
            return image_array[:, :, 0]
        elif image_array.shape[2] == 3:
            # Color image
            return image_array
        elif image_array.shape[2] == 4:
            # RGBA image
            return image_array
    else:
        # Unsupported image type
        raise ValueError("Unsupported image type")

def print_relevant_images(inputText):
    relevantDocs = retriever.get_relevant_documents(inputText, limit=6)
    relevantDocsSplit = split_image_text_types(relevantDocs)
    if "images" in relevantDocsSplit and isinstance(relevantDocsSplit["images"], list):
        st.write("See below for images possibly relevant to this answer.")
        for img_base64 in relevantDocsSplit["images"]:
            image_representation = base64_to_image(img_base64)
            st.image(image_representation)

# Uncomment for running on streamlit
with st.form('my_form'):
    inputText = st.text_area('Enter text:', 'What should be my strategy for planning my project?')
    submitted = st.form_submit_button('Submit')
    if not OPENAI_API_KEY.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and OPENAI_API_KEY.startswith('sk-'):
        st.write_stream(chain_multimodal_rag.stream(inputText))