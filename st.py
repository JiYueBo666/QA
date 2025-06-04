import sys, os
from streamlit_chat import message

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import requests
import time

from LLM.llm import largeModel, chatManager


# 设置页面标题
st.set_page_config(page_title="问答系统", layout="centered")
st.title("混合检索问答系统")

st.write("Streamlit version", st.__version__)

st.markdown(
    """
<style>
    .st-emotion-cache-4oy321 {
        flex-direction: row-reverse;
        text-align: right;
    }
    /* 添加聊天容器的边框样式 */
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)
API_URL = "http://localhost:8000/api/qa"

ai = largeModel()

# 初始化 session_state
if "history_messager" not in st.session_state:
    st.session_state.history_messager = chatManager(
        "你是一个医疗问答助手，"
        "请根据用户的问题和检索结果到的相关诊断历史提供准确的答案，给予用户建议"
    )
    st.session_state.chat_history = []  # 添加聊天历史列表

# 使用 session_state 中的 history_messager
history_messager = st.session_state.history_messager
if "messages" not in st.session_state:
    st.session_state.messages = []

    # 创建输入框
user_query = st.chat_input("请输入你的问题：")

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    # 显示聊天历史
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        else:
            with st.chat_message("user"):
                st.markdown(message["content"])

    # 显示信息这一块，ai和用户是反的，保证左AI右用户
    if user_query:
        # 显示用户输入
        with st.chat_message("assistant"):
            st.markdown(user_query)
        # 将用户消息添加到历史记录
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        try:
            start = time.time()
            with st.status("正在检索问题...", expanded=True) as status:
                # 发起搜索请求
                results = requests.post(API_URL, json={"query": user_query})
                end = time.time()

                status.update(
                    label=f"检索完成，用时:{end - start:.2f}s",
                    state="complete",
                    expanded=True,
                )
        except Exception as e:
            st.error(f"检索过程中发生错误: {e}")
        if results.status_code == 200:
            results = results.json()
            # 记录上下文历史，解析搜索结果
            query = history_messager.parse_search_results(user_query, results)
            history_messager.add_message(role="user", content=query)
            current_messages = history_messager.get_message()

            # 获取流式响应
            with st.status("正在生成回复...", expanded=True) as ai_assistant:
                response_stream = ai.chat(messages=current_messages)
                first_token_received = False
                ai_assistant.update(
                    label="回复生成完成",
                    state="complete",
                    expanded=True,
                )

            def stream_output():
                first_token_received = False
                for chunk in response_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        if not first_token_received:
                            first_token_received = True
                        yield content

            with st.chat_message("user"):
                response = st.write_stream(stream_output())
                # 将助手回复添加到历史记录
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
                # 将完整回复加入历史
                history_messager.add_message(role="assistant", content=response)
    st.markdown("</div>", unsafe_allow_html=True)
