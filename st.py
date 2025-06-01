import streamlit as st
import requests

# 设置页面标题
st.set_page_config(page_title="问答系统", layout="centered")
st.title("混合检索问答系统")

# 定义 API 地址（根据你的 FastAPI 服务地址修改）
API_URL = "http://localhost:8000/api/qa"

# 创建输入框
user_query = st.text_input("请输入你的问题：")

# 创建发送按钮
if st.button("发送"):
    if user_query:
        # 调用 FastAPI 接口
        try:
            response = requests.post(API_URL, json={"query": user_query})
            if response.status_code == 200:
                results = response.json()

                if not results:
                    st.info("未找到相关答案。")
                else:
                    st.subheader("搜索结果：")
                    for idx, ans in enumerate(results, start=1):
                        st.markdown(f"### 结果 {idx}")
                        st.write(f"**标题**: {ans['title']}")
                        st.write(f"**问题**: {ans['ask']}")
                        st.write(f"**答案**: {ans['answer']}")
                        st.write(f"**综合得分**: {ans['final_score']:.4f}")
                        st.markdown("---")
            else:
                st.error(f"请求失败，状态码: {response.status_code}")
        except Exception as e:
            st.error(f"连接错误: {e}")
    else:
        st.warning("请输入一个问题再点击发送！")