from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

from langchain_community.embeddings import XinferenceEmbeddings


def get_qwen2_vl_models():
    llm = ChatOpenAI(temperature=0.2, 
                     top_p=0.2,
                     model="/gemini/code/neb_assistant/models/qwen/Qwen2-VL-7B-Instruct-merged",
                     api_key="XXXX",
                     base_url="http://localhost:8000/v1")
    
    chat = ChatOpenAI(temperature=0.2, 
                      top_p=0.2,
                      model="/gemini/code/neb_assistant/models/qwen/Qwen2-VL-7B-Instruct-merged",
                      api_key="XXXX",
                      base_url="http://localhost:8000/v1")
    
    embed = XinferenceEmbeddings(model_uid="bge-m3",
                                 server_url="http://direct.virtaicloud.com:40202")

    return llm, chat, embed


