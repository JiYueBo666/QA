import sys, os, pprint
import logging
from dotenv import load_dotenv, find_dotenv
from functools import lru_cache

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加到 sys.path
sys.path.append(parent_dir)
from .es_retriver import ElasticRetriver
from .faiss_retriver import FaissRetriver
from utils import timer

_ = load_dotenv(find_dotenv())


class HybridRetriver:
    def __init__(self):
        self.es_retriever = ElasticRetriver()
        self.faiss_retriver = FaissRetriver()
        try:
            self.faiss_retriver.load_index("./faiss")
        except Exception as e:
            logger.warning(f"load faiss failed : {e}")
            self.faiss_retriver = None

    @timer
    @lru_cache
    def hybrid_search(self, query: str, top_k=5):
        vector_search = None
        if self.faiss_retriver is not None:
            # get embedding
            query_emb = self.faiss_retriver.embed_model.embed_query(query)
            vector_search = self.faiss_retriver.search(query_emb, k=top_k)

        text_search = self.es_retriever.search(query, k=top_k)

        hybrid_result = self._merge_result(text_search, vector_search)

        hybrid_result.sort(key=lambda x: x["final_score"], reverse=True)

        return hybrid_result[:5]

    def _merge_result(self, text_results, vector_results=None):
        result_dict = {}
        VECTOR_WEIGHT = float(os.environ["VECTOR_WEIGHT"])
        TEXT_WEIGHT = float(os.environ["TEXT_WEIGHT"])
        # 归一化得分
        self._normalize_score(text_results, key="score")

        if vector_results is not None:
            self._normalize_score(vector_results, key="score")
            for result in vector_results:
                title = result["title"]
                result_dict[title] = {
                    "title": title,
                    "ask": result.get("ask", ""),
                    "answer": result.get("answer", ""),
                    "vector_score": result["score"],
                    "vector_normalized_score": result["score_normalized"],
                    "text_score": 0.0,
                    "text_normalized_score": 0.0,
                    "final_score": result["score_normalized"] * VECTOR_WEIGHT,
                }

        for result in text_results:
            title = result["title"]
            if title in result_dict:
                result_dict[title]["text_score"] = result["score"]
                result_dict[title]["text_normalized_score"] = result["score_normalized"]
                result_dict[title]["final_score"] += (
                    result["score_normalized"] * TEXT_WEIGHT
                )
            else:
                result_dict[title] = {
                    "title": title,
                    "ask": "",
                    "answer": result.get("answer", ""),
                    "vector_score": 0.0,
                    "vector_normalized_score": 0.0,
                    "text_score": result["score"],
                    "text_normalized_score": result["score_normalized"],
                    "final_score": result["score_normalized"] * TEXT_WEIGHT,
                }

        return list(result_dict.values())

    def _normalize_score(self, results, key):
        scores = [res[key] for res in results]
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score if max_score != min_score else 1.0

        for res in results:
            res[f"{key}_normalized"] = (res[key] - min_score) / range_score


from load_data import LoadQA

qa = LoadQA().get_all_qa_pairs()

h = HybridRetriver()
h.faiss_retriver.qa_pairs = qa
r = h.hybrid_search("得了阳痿怎么办")
print(r)
