from langchain_chroma import Chroma
import config_data as config
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class VectorStoreService(object):
    def __init__(self,embedding):
        """

        :param embeddings:嵌入模型传入
        """
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.chroma_collection_name,
            persist_directory=config.chroma_persist_directory,
            embedding_function=embedding)
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})


if __name__ == '__main__':
    retriever=VectorStoreService(OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_BASE"),
        check_embedding_ctx_length=False,
    )).get_retriever()
    res=retriever.invoke("我的体重180斤，尺码推荐")
    print(res)
