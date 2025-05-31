import uvicorn, os, sys
from api.search_api import app

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加到 sys.path
sys.path.append(parent_dir)

DEBUG = False  # 或 False，根据是否是开发环境设置

if __name__ == "__main__":
    uvicorn.run("api.search_api:app", host="0.0.0.0", port=8000, reload=DEBUG)
