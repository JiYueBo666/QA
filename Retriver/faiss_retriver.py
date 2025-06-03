import os
from pathlib import Path
import torch
import faiss
import numpy as np
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
import logging
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

import sys

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加到 sys.path
sys.path.append(parent_dir)
from load_data import LoadQA
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


class EmbeddingModel:
    def __init__(self, default_path="./huggingface/encoder"):
        self.default_path = default_path
        self.download_from_hf()
        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(default_path)
            # 将模型加载到GPU
            self.model = AutoModel.from_pretrained(default_path).to(self.device)
            logger.info(f"模型已加载到设备: {self.device}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise e  # 或者根据需求设为None等处理方式

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def cls_pooling(self, model_output):
        return model_output[0][:, 0]

    def embed_query(self, text):
        try:
            self.model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, padding=True, truncation=True, return_tensors="pt"
                )
                # 将输入数据移动到GPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model_output = self.model(**inputs)
                embeds = self.cls_pooling(model_output).squeeze()
                # 将结果移回CPU进行后续处理
                embeds = embeds.cpu()
                embeds = (
                    torch.nn.functional.normalize(embeds, p=2, dim=0).numpy().tolist()
                )
                return embeds
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return None

    def embed_documents(self, texts):
        """Embed search docs."""
        return [self.embed_query(text) for text in texts]

    def download_from_hf(self):
        # 加载嵌入模型
        try:
            if os.path.exists("./huggingface/encoder"):
                logger.info("embedding model already exist")
                return
            else:
                local_dir = "./huggingface/encoder"
                logger.info(f"ready to download model to path:{local_dir}")
                snapshot_download(
                    repo_id="DMetaSoul/Dmeta-embedding-zh",
                    local_dir=local_dir,
                    proxies={"https": "http://localhost:7890"},
                    max_workers=8,
                )

                logger.info(f"Model is downloading to {local_dir}")
        except Exception as e:
            logging.error(e)


class FaissRetriver:
    def __init__(self):
        self.index = None
        self.qa_pairs = []
        self.embed_model = EmbeddingModel()
        self.vectors = None
        # 检查是否有可用的GPU
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
            # 初始化GPU资源
            self.res = faiss.StandardGpuResources()
            # 配置GPU选项
            self.gpu_options = faiss.GpuClonerOptions()
            self.gpu_options.useFloat16 = True  # 使用半精度浮点数以提高性能
        else:
            logger.info("未检测到GPU，将使用CPU模式")

    def get_vectors(self, qa_pairs):
        texts = [qa_pair.get("title", "") for qa_pair in qa_pairs]

        batch_size = 32
        all_vectors = []

        # 使用 tqdm 正确创建进度条
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
            batch_texts = texts[i : i + batch_size]
            batch_vectors = self.embed_model.embed_documents(batch_texts)
            all_vectors.extend(batch_vectors)

        logger.info(f"Processed {len(texts)} items.")
        self.vectors = all_vectors
        return all_vectors

    def _create_index(self, vectors=None):
        VECTOR_DIM = int(os.environ["EMBED_DIM"])
        print(VECTOR_DIM)
        quantizer = faiss.IndexFlatL2(VECTOR_DIM)

        # 创建CPU版本的索引，减少子量化器数量
        index = faiss.IndexIVFPQ(quantizer, VECTOR_DIM, 500, 32, 8)  # 从64减少到32

        # 如果有GPU可用，将索引转移到GPU
        if self.use_gpu:
            # 修改GPU选项，启用float16查找表以减少内存使用
            self.gpu_options.useFloat16LookupTables = True

            # 将索引转移到GPU，传入GPU选项
            index = faiss.index_cpu_to_gpu(self.res, 0, index, self.gpu_options)
            logger.info("索引已转移到GPU，使用float16查找表和减少的子量化器")

        return index

    def build_index(self, vectors, qa_pairs):
        """
        Build a FAISS index from vectors and store associated QA pairs.
        """
        self.qa_pairs = qa_pairs
        vectors = np.array(vectors).astype("float32")

        self.index = self._create_index()

        if not self.index.is_trained:
            print("[Faiss] Training index...")
            self.index.train(vectors)

        print("[Faiss] Adding vectors to index...")
        self.index.add(vectors)
        self.index.nprobe = min(10, 500)  # default search probe

        print(f"[Faiss] Index built: total vectors = {self.index.ntotal}")

    def search(self, query_vector, k=5):
        """
        Search top-k similar entries for the given query vector.
        """
        if self.index is None:
            raise ValueError("Index not built yet!")

        query_vector = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_vector, k)

        logger.info(f"qa pairs number : [{len(self.qa_pairs)}]")
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append(
                    {
                        "title": self.qa_pairs[idx].get("title", ""),
                        "ask": self.qa_pairs[idx].get("ask", ""),
                        "answer": self.qa_pairs[idx].get("answer", ""),
                        "score": float(1 / (1 + distances[0][i])),  # L2 to similarity
                    }
                )
        return results

    def save_index(self, filepath="./faiss"):
        """
        Save FAISS index to a file.
        """
        if self.index is not None:
            # 如果是GPU索引，需要先转回CPU
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, filepath)
            else:
                faiss.write_index(self.index, filepath)
            logger.info(f"[Faiss] Index saved to {filepath}")

    def load_index(self, filepath):
        """
        Load FAISS index from a file.
        """
        # 先加载到CPU
        cpu_index = faiss.read_index(filepath)

        # 如果有GPU可用，将索引转移到GPU
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
            logger.info(f"[Faiss] Index loaded from {filepath} to GPU")
        else:
            self.index = cpu_index
            logger.info(f"[Faiss] Index loaded from {filepath} to CPU")


# if __name__ == "__main__":
#     data = LoadQA().get_all_qa_pairs()

#     fais = FaissRetriver()
    # vectors = fais.get_vectors(data)
    # fais.build_index(vectors, data)
    # fais.save_index()

    # fais.load_index("./faiss")
    # fais.qa_pairs = data
    # vec = fais.embed_model.embed_query("得了艾滋病感觉人生完了")
    # r = fais.search(vec)
    # print(r)
    # print(type(r))
