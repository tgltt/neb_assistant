from dotenv import load_dotenv
load_dotenv(".qwen")

from langchain_community.chat_models import ChatTongyi
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TextIteratorStreamer

# DEFAULT_CKPT_PATH = 'models/qwen/Qwen2-VL-7B-Instruct'
DEFAULT_CKPT_PATH = 'saves/Qwen2VL-7B-Chat/lora/train_sft_2024-09-04-02-02-30'

def get_qwen_models(llm_model="qwen-max", chat_model="qwen-max", embedding_model="text-embedding-v1"):
    llm = Tongyi(model=llm_model,
                 top_p=0.8,
                 temperature=0.1,
                 max_tokens=1024)

    chat = ChatTongyi(model=chat_model,
                      top_p=0.8,
                      temperature=0.1,
                      max_tokens=1024)
    
    embedding = DashScopeEmbeddings(model=embedding_model)
    
    return llm, chat, embedding


# def get_qwen_vl_7B_chat_model():
#     model = Qwen2VLForConditionalGeneration.from_pretrained(DEFAULT_CKPT_PATH, device_map='auto')

#     processor = AutoProcessor.from_pretrained(DEFAULT_CKPT_PATH)
#     return model, processor

def get_qwen_vl_7B_models():
    from langchain_community.llms import Xinference

    chat = Xinference(model_uid="qwen2-vl-instruct_1",
                      server_url="http://direct.virtaicloud.com:40202")
    
    return None, chat, None