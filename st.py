import sys, os
from io import StringIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import requests
import time
from openai import OpenAI

from LLM.llm import largeModel, chatManager


# 设置页面标题
st.set_page_config(page_title="问答系统", layout="centered")
st.title("混合检索问答系统")

# 定义 API 地址（根据你的 FastAPI 服务地址修改）
API_URL = "http://localhost:8000/api/qa"

ai = largeModel()

# 初始化 session_state
if "history_messager" not in st.session_state:
    st.session_state.history_messager = chatManager(
        "你是一个医疗问答助手，"
        "请根据用户的问题和检索结果到的相关诊断历史提供准确的答案，给予用户建议"
    )

# 使用 session_state 中的 history_messager
history_messager = st.session_state.history_messager

# 创建输入框
user_query = st.chat_input("请输入你的问题：")
if user_query:
    print(f"当前的上下文:{history_messager.get_message()}")
    try:
        start = time.time()
        with st.status("正在检索问题...", expanded=True) as status:
            results = requests.post(API_URL, json={"query": user_query})
            end = time.time()
            status.update(
                label=f"检索完成，用时:{end - start:.2f}s",
                state="complete",
                expanded=True,
            )

        if results.status_code == 200:
            results = results.json()

            # 创建一个占位符用于显示“生成中”和“生成完成”的提示
            spinner_placeholder = st.empty()
            spinner_placeholder.info("🧠 正在生成回复...")

            # 记录上下文历史，解析搜索结果
            query = history_messager.parse_search_results(user_query, results)
            history_messager.add_message(role="user", content=query)
            current_messages = history_messager.get_message()
            print(f"接受用户输入后的上下文:{current_messages}")
            # 获取流式响应
            response_stream = ai.chat(messages=current_messages)
            full_response = []
            first_token_received = False

            # 定义一个生成器函数供 st.write_stream 使用
            def stream_output():
                first_token_received = False
                for chunk in response_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        if not first_token_received:
                            first_token_received = True
                            spinner_placeholder.success("✅ 生成完成")
                        full_response.append(content)
                        yield content

            st.write_stream(stream_output())

            # 此时 full_response 已经被填充
            full_content = "".join(full_response)
            # 回复加入历史
            history_messager.add_message(role="assistant", content=full_content)

        else:
            st.error(f"请求失败，状态码: {results.status_code}")
    except Exception as e:
        st.error(f"连接错误: {e}")
