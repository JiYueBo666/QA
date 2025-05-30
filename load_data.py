import pandas as pd
from typing import List, Dict

# from QA.config import MEDICAL_DATASET_DIR, TEST_NUM
import os
import logging
from glob import glob


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadQA:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.data = None
        self.load_csv(self.data_dir)

    def load_csv(self, folder_path: str):
        all_csv_files = glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
        if not all_csv_files:
            logger.warning(f"No csv found in {folder_path}")
            return
        combined_df = pd.concat(
            [pd.read_csv(file) for file in all_csv_files], ignore_index=True
        )
        print(f"Load QA data total ：{len(combined_df)}")

        self.data = combined_df

    def get_all_qa_pairs(self) -> List[Dict[str, str]]:
        return self.data[["department", "title", "ask", "answer"]].to_dict("records")
