import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

md5_path = os.path.join(BASE_DIR, "data", "md5.txt")



# Chroma
chroma_collection_name='rag'
chroma_persist_directory = os.path.join(BASE_DIR, "chroma_db")

# Splitter
chunk_size=1000
chunk_overlap=100
separators=["\n\n", "\n", ".","!", "?",","," ",""]
max_split_chunk_size=1000


similarity_threshold=2

#model
embedding_model="text-embedding-v4"
chat_model='qwen3.8-max'
