"""
知识库
"""
import os
from datetime import datetime

from langchain_chroma import Chroma
import config_data as config
import hashlib
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv(override=True)

def check_md5(md5_str:str):
    """检查传入的md5字符串是否处理过了"""
    if not os.path.exists(config.md5_path):
        os.makedirs(os.path.dirname(config.md5_path), exist_ok=True)
        open(config.md5_path, "w").close()
        return False
    with open(config.md5_path,"r") as f:
        md5_list=f.readlines()
        for line in md5_list:
            if line.strip() == md5_str:
                return True
        return False
def save_md5(md5_str:str):
    """将传入md5字符串保存在文件"""
    with open(config.md5_path,"a") as f:
        f.write(md5_str+"\n")

def get_str_md5(str_data:str):
    """将传入字符串改为md5字符串"""
    md5_obj = hashlib.md5()
    md5_obj.update(str_data.encode("utf-8"))
    return md5_obj.hexdigest()

class KnowledgeBaseService(object):
    def __init__(self):
        #如果文件夹不存在则创建，存在则跳过
        os.makedirs(config.chroma_persist_directory, exist_ok=True)
        self.chroma=Chroma(
            collection_name=config.chroma_collection_name,
            embedding_function=OpenAIEmbeddings(
            model="text-embedding-v4",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE"),
            check_embedding_ctx_length=False,
    ),
            persist_directory=config.chroma_persist_directory,
        )
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,#分片大小
            chunk_overlap=config.chunk_overlap,#分片重叠
            separators=config.separators,#分片分隔符
            length_function=len,#分片长度函数
        )

    def upload_by_str(self,data,filename):
        """将字符串数据向量化,保存在向量数据库中"""
        md5_hex=get_str_md5(data)
        if check_md5(md5_hex):
            print("[跳过]文件已被处理")
            return "[跳过]文件已被处理"
        if len(data)>config.max_split_chunk_size:
            knowledge_chunks=self.spliter.split_text(data)
        else:
            knowledge_chunks=[data]
        metadata={
                "filename": filename,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "author":"小李"
                    }
        self.chroma.add_texts(knowledge_chunks, metadata=[metadata for _ in knowledge_chunks])
        save_md5(md5_hex)
        print("[成功]文件已保存到向量库")
        return "[成功]文件已保存到向量库"
