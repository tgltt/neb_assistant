import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image

from io import BytesIO  
from pathlib import Path

from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_document import load_document

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from rag import chat_rag

import streamlit as st

from chromadb import Client
from chromadb import Settings
import chromadb

from chroma_helper import ChromaDBHelper

import tempfile
from utils import get_qwen2_vl_models

import base64

CHROMA_COLLECTION_NAME = "neb_rag"

ENABLE_RAG_LABEL = "启用RAG"
DISABLE_RAG_LABEL = "禁用RAG"

IMAGE_MAX_WIDTH = 448

@st.cache_resource
def get_model():
    return get_qwen2_vl_models()

@st.cache_resource
def get_chroma_db():
    chroma_helper = ChromaDBHelper(host="localhost", port=8002)
    store = chroma_helper.get_chroma(CHROMA_COLLECTION_NAME, embed)
    return chroma_helper, store

def mkdir_if_necessary(path):
    if not os.path.exists(path):
        print(f"创建目录{path}")
        os.makedirs(name=path, exist_ok=True)

def whether_enable_rag(selection):
    return selection == ENABLE_RAG_LABEL if len(selection) > 0 else False

def update_store(current_directory, uploaded_file, store):
    st.text(f"更新向量库, file={uploaded_file.name}")

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    file_name = uploaded_file.name
    save_path = current_directory / temp / file_name

    mkdir_if_necessary(str(save_path.parent))

    # 使用BytesIO来读取上传的文件内容
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    docs = load_document(file_path=save_path, file_extension=file_extension)

    spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=128)
    docs = spliter.split_documents(documents=docs)
    # Chroma数据库的初始化，并在当前路径下创建一个名为chroma_data的目录，进行数据库持久化储存
    docs = filter_complex_metadata(docs)

    batch_size = 6
    # 批量插入数据到Chroma数据库
    for idx in range(0, len(docs), batch_size):
        print(idx)
        store.add_documents(documents=docs[idx: idx + batch_size])

def on_upload_chat_file_changed():
    st.session_state["on_upload_chat_file_changed"] = True

def deal_upload_chat_file_changed():
    print("on_upload_chat_file_changed, " + str(uploaded_chat_files))
    if "chat_files_uploaded" not in st.session_state:
        st.session_state["chat_files_uploaded"] = []

    session_chat_files_uploaded = st.session_state["chat_files_uploaded"]
    if uploaded_chat_files is None:
        session_chat_files_uploaded.clear()

    for uploaded_chat_file in uploaded_chat_files:
        if uploaded_chat_file not in session_chat_files_uploaded:
            session_chat_files_uploaded.append(uploaded_chat_file)
            st.session_state.messages.append(
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": local_image_to_data_url(uploaded_chat_file)}}]})

    st.session_state["on_upload_chat_file_changed"] = False

def is_image(type):
    return type.lower() in ["image", "image_url"]

def clear_history():
    if "messages" in st.session_state:
        st.session_state.messages.clear()

# Function to encode a local image into data URL
def local_image_to_data_url(image_file, mime_type=None):
    # print(f"----------------->image_file: {dir(image_file)}")
    
    image_raw = Image.open(image_file)
    if image_raw is None :
        print(f"{image_file} open failed")
        return
    
    # 设置图片的新尺寸
    width, height = image_raw.size
    new_width = min(width, IMAGE_MAX_WIDTH)  # 最大宽度是IMAGE_MAX_WIDTH像素
    new_height = int((height * new_width) / width)
    
    # 调整图片大小
    resized_image = image_raw.resize((new_width, new_height))
    
    # Default to png
    if mime_type is None:
        mime_type = 'image'

    io_bytes = BytesIO()
    resized_image.save(io_bytes, format=image_file.type.split("/")[-1])
    
    base64_encoded_data = base64.b64encode(io_bytes.getvalue()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

st.set_page_config(page_title="RAG知识注入和加载", page_icon="📖", layout='wide')

_, chat, embed = get_model()
chroma_helper, store = get_chroma_db()

# 上传文件
# 侧边栏显示
st.sidebar.write('### RAG知识系统')

expander = st.sidebar.expander("RAG功能设置")

current_directory = Path.cwd()
RAG_selection = expander.radio(
    '请选择是否启用RAG知识:',
    (ENABLE_RAG_LABEL, DISABLE_RAG_LABEL)
)

if whether_enable_rag(RAG_selection):
    temp = "tempfile"
    # 使用函数
    uploaded_rag_files = expander.file_uploader("上传文件，进行新知识注入",
                                                accept_multiple_files=True,
                                                type=['txt', 'docx', 'md', 'pdf', 'csv','xlsx','pptx','html'])
    docs = []
    save_path = ""

    # 判断文件是否上传，并进行数据加载
    if len(uploaded_rag_files) > 0:
        for uploaded_rag_file in uploaded_rag_files:
            # 获取文件后缀名
            update_store(current_directory, uploaded_rag_file, store)

st.sidebar.write("### 聊天功能")
uploaded_chat_files = st.sidebar.file_uploader("上传图片",
                                               accept_multiple_files=True,
                                               type=['png', 'jpg', 'jpeg'],
                                               on_change=on_upload_chat_file_changed)
st.sidebar.button("清空历史", on_click=clear_history)

if "on_upload_chat_file_changed" in st.session_state \
    and st.session_state["on_upload_chat_file_changed"] \
    and uploaded_chat_files is not None:
    deal_upload_chat_file_changed()

retriever = store.as_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 多模态新能源电池研发助手")

st.chat_message("assistant").write("您好，我是您的研发助手，有什么可以帮您？")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        sub_msgs = message["content"]
        for sub_msg in sub_msgs:
            if is_image(sub_msg["type"]):
                st.image(sub_msg["image_url"]["url"], width=IMAGE_MAX_WIDTH)
                continue
            st.markdown(sub_msg["text"])

info = st.chat_input()
if info is not None and len(info) > 0:
    st.chat_message('user').markdown(info)
    st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": info}]})

    with st.spinner("加载中..."):
        with st.chat_message("assistant"):
            output_placeholder = st.empty()
            for response in chat_rag(model=chat,
                                     retriever=retriever,
                                     query=info,
                                     history=st.session_state.messages,
                                     enable_rag_flag=whether_enable_rag(RAG_selection)):
                output_placeholder.text(response)
            
            st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
