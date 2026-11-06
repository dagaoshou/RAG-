"""
基于Streamlit完成web网页上传服务
"""
import time

import streamlit as st
from knowledge_base import KnowledgeBaseService
st.title("知识库更新服务")
#file_uploader
upload_file = st.file_uploader(
    "请上传TXT文件",
    type=["txt"],
    accept_multiple_files=False, # 默认为False，即一次只能上传一个文件
)
#session_state是一个字典
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()
if upload_file is not None:
    #提取文件信息
    file_name = upload_file.name
    file_type = upload_file.type
    file_size = upload_file.size/1024

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type}| 大小：{file_size:.2f}KB")
    #get_value:
    text=upload_file.getvalue().decode("utf-8")

    with st.spinner("上传入知识库中..."):
        time.sleep(1)
        result=st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)
