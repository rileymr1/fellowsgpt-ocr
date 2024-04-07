import io
import re
import base64
import streamlit as st
import certifi
import os
import numpy as np


# from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.storage import InMemoryStore

from dotenv import load_dotenv
os.environ.clear()
load_dotenv()

# Get secret keys from environment variables
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
MONGODB_CONN_STRING=st.secrets["MONGODB_CONN_STRING"]
DB_NAME=st.secrets["DB_NAME"]
VECTOR_COLLECTION_NAME=st.secrets["VECTOR_COLLECTION_NAME"]
KEYVALUE_COLLECTION_NAME=st.secrets["KEYVALUE_COLLECTION_NAME"]
VECTOR_INDEX_NAME=st.secrets["VECTOR_INDEX_NAME"]



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

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


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\xFF\xD8\xFF": "jpeg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False
    

def img_prompt_func(data_dict): ## data_dict is the same `{context: ___, question: ____}` that is defined in the chain
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a healthcare consultant tasking with providing training to newly hired early career healthcare consultant trainees called 'Fellows.' .\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide advice related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1920, 1080))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def get_relevant_docs_ids(docs):
    doc_ids = []
    for doc in docs:
        doc_ids.append(doc.metadata['doc_id'])
    return doc_ids

def get_relevant_docs_contents(doc_ids):
    rel_docs_contents = []
    for doc_id in doc_ids:
        query = {"doc_id": doc_id}
        rel_doc_content_full = raw_contents.find_one(query)
        rel_doc_content_formatted = {key: value for key, value in rel_doc_content_full.items() if key != '_id' and key != 'doc_id'}
        rel_docs_contents.append(rel_doc_content_formatted)
    return rel_docs_contents

def format_context(rel_docs_contents):
    b64_images = []
    raw_texts = []
    img_names = []
    for doc_content in rel_docs_contents:
        # Resize & append images
        base64_img = doc_content['base64_img']
        image = resize_base64_image(base64_img, size=(1920, 1080))
        b64_images.append(image)

        # Append raw text
        raw_texts.append(doc_content['raw_text'])

        # Append img_names
        img_names.append(doc_content['img_name'])
    return {"images": b64_images, "texts": raw_texts, "img_names": img_names}


# Write relevant documents to output first, then continue on with the chain
def print_relevant_images(data_dict):
    st.write("See below for images possibly relevant to this answer.")
    if data_dict["context"]["images"]:
        for index, base64_img in enumerate(data_dict["context"]["images"]):
            image_representation = base64_to_image(base64_img)
            st.image(image_representation)
            st.write("This image is from " + data_dict["context"]["img_names"][index])
            # print (image_representation) # Uncomment if not running on streamlit
    st.write("Generating text response...")
    # Pass the parameter unchanged to the next link in chain       
    return data_dict


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", openai_api_key=OPENAI_API_KEY)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(get_relevant_docs_ids) | RunnableLambda(get_relevant_docs_contents) | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }   
        | RunnableLambda(print_relevant_images)
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain

# Initialize MongoDB clients
client = MongoClient(MONGODB_CONN_STRING, tlsCAFile=certifi.where())
mongoDB = client[DB_NAME]
vector_collection = client[DB_NAME][VECTOR_COLLECTION_NAME]
kv_collection = client[DB_NAME][KEYVALUE_COLLECTION_NAME]

store = InMemoryStore()

print ("MongoDB connected: ", mongoDB)

raw_contents = kv_collection

vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_CONN_STRING,
    DB_NAME + "." + VECTOR_COLLECTION_NAME,
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, disallowed_special=()), # embeddings, # OpenAIEmbeddings(disallowed_special=()),
    index_name=VECTOR_INDEX_NAME
)

# Create a vectorstore-backed retriever
retriever = vectorstore.as_retriever()

# Create RAG chain
chain_multimodal_rag = multi_modal_rag_chain(retriever)

# Uncomment for easy debugging
# print(chain_multimodal_rag.invoke("What should be my strategy for planning my project?"))

