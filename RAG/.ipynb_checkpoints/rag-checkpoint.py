import copy

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.messages import HumanMessage

from transformers import TextIteratorStreamer


def format_docs(docs):
    return ("\n\n".join(doc.page_content for doc in docs),
            "\n\n".join(f"{idx + 1}、{doc.metadata['source']}，第{doc.metadata['page'] + 1}页，总共{doc.metadata['total_pages']}页" for idx, doc in enumerate(docs))
            )

# def chat_rag(model, retriever, query, history, enable_rag_flag=True):
#     messages = copy.deepcopy(history)
    
#     if enable_rag_flag:
#         retriever_chain = retriever | format_docs
#         context, references = retriever_chain.invoke(input=query)

#         prompt = """You are an assistant for question-answering tasks. Use the following pieces 
#                   of retrieved context to answer the question. 
#                   Use three sentences maximum and keep the answer detailed.
#                   Question: {question} 
#                   Context: {context} 
#                   Answer:"""
#         # RAG 链
#         # rag_chain = prompt | model | StrOutputParser()
#         # result = rag_chain.invoke(input={"context": context, "question": query})

#         content = []
#         content.append({"type": "text", "text": prompt.format(context=context, question=query)})
#         messages.append({"role": "user", "content": content})
        
#         result = model.invoke(input=messages).content
        
#         if len(references) > 0:
#             result = f"{result}\n\n参考以下资源:\n\n{references}"    
#     else:
#         content = []
#         messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        
#         result = model.invoke(input=messages).content

#     return result

def chat_rag(model, retriever, query, history, enable_rag_flag=True):
    messages = copy.deepcopy(history)

    result = ""
    if enable_rag_flag:
        retriever_chain = retriever | format_docs
        context, references = retriever_chain.invoke(input=query)

        prompt = """You are an assistant for question-answering tasks. Use the following pieces 
                  of retrieved context to answer the question. 
                  Use three sentences maximum and keep the answer detailed.
                  Question: {question} 
                  Context: {context} 
                  Answer:"""
        # RAG 链
        # rag_chain = prompt | model | StrOutputParser()
        # result = rag_chain.invoke(input={"context": context, "question": query})

        content = []
        content.append({"type": "text", "text": prompt.format(context=context, question=query)})
        messages.append({"role": "user", "content": content})

        
        stream = model.stream(input=messages)
        for chunk in stream:
            result += chunk.content
            yield result if len(references) <= 0 else f"{result}\n\n参考以下资源:\n\n{references}"

        if len(references) > 0:
            result = f"{result}\n\n参考以下资源:\n\n{references}"
    else:
        content = []
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        
        stream = model.stream(input=messages)
        
        for chunk in stream:
            result += chunk.content
            yield result

    return result