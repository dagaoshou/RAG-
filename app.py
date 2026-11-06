from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from rag import CHAT_HISTORY_DIR, RagService, get_history


st.set_page_config(
    page_title="RAG本地知识库",
    page_icon=":material/chat_bubble:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    updated_at: datetime
    message_count: int
    preview: str


@st.cache_resource(show_spinner="正在加载本地知识库...")
def load_rag_service() -> RagService:
    return RagService()


def message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def list_sessions() -> list[SessionSummary]:
    history_dir = Path(CHAT_HISTORY_DIR)
    history_dir.mkdir(parents=True, exist_ok=True)
    sessions = []

    for path in history_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue

        try:
            messages = get_history(path.name).messages
        except (OSError, ValueError, TypeError):
            messages = []

        last_message = message_text(messages[-1]).strip() if messages else ""
        preview = last_message.replace("\n", " ")[:36] if last_message else "空会话"
        sessions.append(
            SessionSummary(
                session_id=path.name,
                updated_at=datetime.fromtimestamp(path.stat().st_mtime),
                message_count=len(messages),
                preview=preview,
            )
        )

    return sorted(sessions, key=lambda item: item.updated_at, reverse=True)


def activate_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id


def create_session_id() -> str:
    return f"chat_{datetime.now():%Y%m%d_%H%M%S_%f}"


def session_label(session: SessionSummary) -> str:
    return f"{session.updated_at:%m-%d %H:%M} · {session.preview}"


sessions = list_sessions()
st.session_state.setdefault(
    "active_session_id",
    sessions[0].session_id if sessions else "user_01",
)

active_session_id = st.session_state.active_session_id
known_session_ids = {session.session_id for session in sessions}
if active_session_id not in known_session_ids:
    sessions.insert(
        0,
        SessionSummary(
            session_id=active_session_id,
            updated_at=datetime.now(),
            message_count=0,
            preview="新会话",
        ),
    )

with st.sidebar:
    st.title(":material/forum: 会话")
    if st.button(
        "新建会话",
        icon=":material/add_comment:",
        type="primary",
        width="stretch",
    ):
        st.session_state.active_session_id = create_session_id()
        st.rerun()

    st.caption(f"当前会话：{active_session_id}")
    st.subheader("历史记录")

    with st.container(height=320, border=False):
        for session in sessions:
            is_active = session.session_id == active_session_id
            st.button(
                session_label(session),
                key=f"session_{session.session_id}",
                type="primary" if is_active else "secondary",
                width="stretch",
                help=(
                    f"会话 ID：{session.session_id}\n"
                    f"消息数：{session.message_count}\n"
                    f"更新时间：{session.updated_at:%Y-%m-%d %H:%M:%S}"
                ),
                on_click=activate_session,
                args=(session.session_id,),
            )

    if st.button(
        "清空当前会话",
        icon=":material/delete_sweep:",
        width="stretch",
    ):
        get_history(active_session_id).clear()
        st.rerun()


st.title("RAG本地知识库")
st.caption("基于本地资料检索，并通过大模型流式生成回答。")

try:
    rag_service = load_rag_service()
except Exception as exc:
    st.error(
        "知识库加载失败，请检查环境变量和向量数据库配置。",
        title="无法启动 RAG 服务",
        icon=":material/error:",
    )
    st.exception(exc)
    st.stop()

try:
    history_messages = get_history(active_session_id).messages
except (OSError, ValueError, TypeError) as exc:
    st.error("当前会话历史读取失败。", title="历史记录错误", icon=":material/error:")
    st.exception(exc)
    history_messages = []

if not history_messages:
    st.info(
        "可以直接询问尺码推荐、服装搭配等知识库问题。",
        title="开始一个新问题",
        icon=":material/lightbulb:",
    )

for message in history_messages:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        continue

    with st.chat_message(role):
        st.markdown(message_text(message))

if prompt := st.chat_input(
    "输入您的问题...",
    submit_mode="disable",
):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                rag_service.chain.stream(
                    {"input": prompt},
                    config={"configurable": {"session_id": active_session_id}},
                ),
                cursor="▌",
            )
            if isinstance(response, str) and not response.strip():
                st.warning(
                    "模型没有返回文本内容，请换一种问法后重试。",
                    icon=":material/warning:",
                )
        except Exception as exc:
            st.error(
                "回答生成失败，请稍后重试。",
                title="请求未完成",
                icon=":material/error:",
            )
            st.exception(exc)
