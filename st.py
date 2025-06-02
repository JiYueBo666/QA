import streamlit as st
import requests
import time
from openai import OpenAI

# 设置页面标题
st.set_page_config(page_title="问答系统", layout="centered")
st.title("混合检索问答系统")

# 定义 API 地址（根据你的 FastAPI 服务地址修改）
API_URL = "http://localhost:8000/api/qa"


def generate_response(query, results):

    titles = []
    questions = []
    answers = []
    scores = []

    client = OpenAI(
        api_key="sk-T4T1ktwoe50DKBY6C79c3f0903Bb4dC798F5E7069aF966B7",
        base_url="https://api.openai-next.com/v1",
    )

    for idx, ans in enumerate(results, start=1):
        titles.append(ans["title"])
        questions.append(ans["ask"])
        answers.append(ans["answer"])
        scores.append(ans["final_score"])

    prompt_title = "下面是一些搜索出来的其他病人的提问和回复结果"

    formatted_results = ""
    for i in range(len(titles)):
        formatted_results += f"""示例 {i+1}:
    标题: {titles[i]}
    问题: {questions[i]}
    答案: {answers[i]}
    #
    """

    print(formatted_results)

    user_prompt = """
    下面是用户的问题:{query}
    请你开始回答。

"""

    final_prompt = prompt_title + formatted_results + user_prompt.format(query=query)

    sys_prompt = (
        "你是一个医疗问答助手，"
        "请根据用户的问题和检索结果提供准确的答案，给予用户建议"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": final_prompt},
        ],
    )

    return completion


# 创建输入框
user_query = st.chat_input("请输入你的问题：")
if user_query:
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

            # 获取流式响应
            response_stream = generate_response(user_query, results)

            # 定义一个生成器函数供 st.write_stream 使用
            def stream_output():
                first_token_received = False
                for chunk in response_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        if not first_token_received:
                            first_token_received = True
                            spinner_placeholder.success("✅ 生成完成")
                        yield content

            # 使用 write_stream 显示流式输出
            st.write_stream(stream_output())

        else:
            st.error(f"请求失败，状态码: {results.status_code}")
    except Exception as e:
        st.error(f"连接错误: {e}")
