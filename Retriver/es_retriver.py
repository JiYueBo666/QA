from elasticsearch import Elasticsearch
from dotenv import load_dotenv, find_dotenv
import os, logging, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加到 sys.path
sys.path.append(parent_dir)

from load_data import LoadQA
from utils import timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())

port = os.environ["ES_PORT"]
url = os.environ["ES_HOST"]
index_name = os.environ["ES_IDX_NAME"]


class ElasticRetriver:
    def __init__(self) -> None:
        try:
            self.es = Elasticsearch(f"http://{url}:{port}")

            if self.es.ping():
                logger.info("elasticsearch connected")
            else:
                logger.info("elasticsearch connection failed")
        except Exception as e:
            logger.info(f"create elasticsearch failed:{e}")
            self.es = None
        self.index_name = index_name

    @timer
    def create_index(self):
        """
        创建ES索引
        """
        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "question": {"type": "text", "analyzer": "ik_max_word"},
                    "answer": {"type": "text", "analyzer": "ik_max_word"},
                    "agent": {"type": "keyword"},
                }
            }
        }

        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=mapping)

    @timer
    def index_qa_pairs(self, qa_pairs):
        for qa in qa_pairs:
            doc = {
                "ask": qa["ask"],
                "title": qa["title"],
                "answer": qa["answer"],
                "department": qa["department"],
            }
            self.es.index(index=self.index_name, body=doc)
        self.es.indices.refresh(index=self.index_name)

    @timer
    def search(self, query, k=5):
        search_body = {"query": {"match": {"title": query}}, "size": k}
        response = self.es.search(index=self.index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append(
                {
                    "title": hit["_source"]["title"],
                    "answer": hit["_source"]["answer"],
                    "score": hit["_score"],
                    "normalized_score": hit["_score"] / response["hits"]["max_score"],
                }
            )

        return results

    def delete_index(self):
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)


# if __name__ == "__main__":
#     data = LoadQA()
#     qa_pairs = data.get_all_qa_pairs()

#     es = ElasticRetriver()

    # #只创建一次索引（如果不存在）
    # es.create_index()

    # es.index_qa_pairs(qa_pairs)

    # # 搜索测试
    # results = es.search("阳痿怎么办")
    # print(results)
    # print(type(results))
