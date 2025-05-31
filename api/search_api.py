import sys, os, pprint
import logging
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加到 sys.path
sys.path.append(parent_dir)
from Retriver.hybridsearch import HybridRetriver
from load_data import LoadQA
from utils import timer

_ = load_dotenv(find_dotenv())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import asyncio

app = FastAPI(
    title="chatbot system",
    description="混合检索问答",
    version="1.0.0",
)

# 将全局变量初始化移到函数中
retriever = None
qa_pairs = None

def init_resources():
    global retriever, qa_pairs
    if retriever is None:
        logger.info("初始化资源...")
        retriever = HybridRetriver()
        qa_pairs = LoadQA().get_all_qa_pairs()
        retriever.faiss_retriver.qa_pairs = qa_pairs
        logger.info("资源初始化完成")

@app.on_event("startup")
async def startup_event():
    init_resources()


class Query(BaseModel):
    query: str = ""


class Answer(BaseModel):
    title: str
    ask: str
    answer: str
    vector_score: float
    vector_normalized_score: float
    text_score: float
    text_normalized_score: float
    final_score: float


# 删除第一个实现，只保留第二个异步实现
@app.post("/api/qa", response_model=List[Answer])
async def get_answer(query: Query):
    """
    Main-API: get_answer (async)
    """
    global retriever
    if retriever is None:
        init_resources()
        
    loop = asyncio.get_event_loop()
    # 使用线程池执行同步方法，避免阻塞事件循环
    results = await loop.run_in_executor(None, retriever.search, query.query)

    return [
        Answer(
            title=res.get("title", ""),
            ask=res.get("ask", ""),
            answer=res.get("answer", ""),
            vector_score=res.get("vector_score", 0.0),
            vector_normalized_score=res.get("vector_normalized_score", 0.0),
            text_score=res.get("text_score", 0.0),
            text_normalized_score=res.get("text_normalized_score", 0.0),
            final_score=res.get("final_score", 0.0),
        )
        for res in results
    ]
