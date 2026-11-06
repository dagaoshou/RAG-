import json
import os.path

from langchain_core.messages import message_to_dict,messages_from_dict
from langchain_core.chat_history import BaseChatMessageHistory

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id,storage_path):
        self.session_id=session_id
        self.storage_path=storage_path
        # 创建文件路径
        self.file_path=os.path.join(self.storage_path,self.session_id)
        #确保文件夹存在
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_message(self, message):
        all_messages = list(self.messages)
        all_messages.append(message)

        new_messages = [message_to_dict(msg) for msg in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)
    @property
    def messages(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages = json.load(f)
                return messages_from_dict(messages)
        except FileNotFoundError:
            return []
    def clear(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)


