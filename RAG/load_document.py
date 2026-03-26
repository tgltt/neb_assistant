from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
# import os
def load_document(file_path,file_extension):  
    """  
    根据文件扩展名加载文档内容。  
    :param file_path: 文件的路径  
     
    """  
        
    # 文件扩展名判断和数据加载  
    if file_extension in ['.txt']:  
        loader = TextLoader(file_path,encoding="utf-8")
        return loader.load()
   
    if file_extension in ['.docx']:
        loader = UnstructuredWordDocumentLoader(file_path,mode="single")
        return loader.load()
  
    if file_extension in ['.md']:
        loader = UnstructuredMarkdownLoader(file_path,mode="single")
        return loader.load()
 
    if file_extension in ['.pdf']:
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    if file_extension in ['.csv']:
        loader = CSVLoader(file_path)
        return loader.load()

    if file_extension in ['.xlsx']:
        loader = UnstructuredExcelLoader(file_path,encoding="utf-8")
        return loader.load()

    if file_extension in ['.pptx']:
        loader = UnstructuredPowerPointLoader(file_path,encoding="utf-8",mode="single")
        return loader.load()

    if file_extension in ['.html']:
        loader = UnstructuredHTMLLoader(file_path,mode= "single")
        return loader.load()
    else:
        return(f"上传文件格式有问题: {file_extension}")
