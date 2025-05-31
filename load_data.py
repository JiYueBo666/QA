import pandas as pd
from typing import List, Dict

# from QA.config import MEDICAL_DATASET_DIR, TEST_NUM
import os, sys
import logging
from glob import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加到 sys.path
sys.path.append(parent_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import timer


class LoadQA:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.data = None
        self.load_csv(self.data_dir)

    @timer
    def load_csv(self, folder_path: str):
        all_csv_files = glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
        if not all_csv_files:
            logger.warning(f"No csv found in {folder_path}")
            return
        combined_df = pd.concat(
            [pd.read_csv(file) for file in all_csv_files], ignore_index=True
        )
        logger.info(f"Load QA data total ：{len(combined_df)}")

        self.data = combined_df

    def get_all_qa_pairs(self) -> List[Dict[str, str]]:
        return self.data[["department", "title", "ask", "answer"]].to_dict("records")
