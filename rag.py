import os
from dotenv import load_dotenv
from vector_stores import VectorStoreService
import config_data as config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from file_history_store import FileChatMessageHistory
from operator import itemgetter
load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(BASE_DIR), "chat_history")

def get_history(session_id):
    return FileChatMessageHistory(session_id, CHAT_HISTORY_DIR)
def paser_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagService(object):
    def __init__(self):
        self.vector_service=VectorStoreService(
            embedding=OpenAIEmbeddings(
                model=config.embedding_model,
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv("DASHSCOPE_API_BASE"),
                check_embedding_ctx_length=False,
            )
        )
        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的资料为主,简介专业回答用户的问题，参考资料{context}"),
                ("system","并且我提供用户历史信息如下："),
                MessagesPlaceholder("history"),
                ("user","请回答用户问题{input}")

            ]
        )
        self.chat_model=init_chat_model(
            model=config.chat_model,
            model_provider="openai",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE")
        )

        self.chain=self.__get_chain()

    def __get_chain(self):
        def format_document(docs: list[Document])->str:
            if not docs:
                return "无相关参考资料"
            content=""
            for doc in docs:
                content+=f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return content
        retriever=self.vector_service.get_retriever()
        chain = (
                RunnablePassthrough.assign(
                    context=itemgetter("input") | retriever | format_document
                )
                | self.prompt_template
                | paser_prompt
                | self.chat_model
                | StrOutputParser()
        )

        conservation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_key="input",
            history_messages_key="history"

        )
        return conservation_chain
if __name__ == '__main__':
    session_config={
        "configurable":{
            "session_id": "user_01"
        }
    }
    res=RagService().chain.invoke({"input":"我身高170cm，尺码推荐"},session_config)
    print(res)
