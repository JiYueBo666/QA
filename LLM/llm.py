from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os, sys, logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class chatManager:
    def __init__(self, system_prmopt="You are a helpful assistant."):
        self.system_prompt = system_prmopt
        self.messages = [
            {"role": "system", "content": system_prmopt},
        ]

    def get_message(self):
        return self.messages

    def add_message(self, role, content):
        item = {"role": role, "content": content}
        self.messages.append(item)

    def parse_search_results(self, query, results):
        titles = []
        questions = []
        answers = []
        scores = []
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

        user_prompt = """
        下面是用户的问题:{query}
        请你开始回答。

        """
        final_prompt = user_prompt.format(query=query)
        return final_prompt


class largeModel:
    def __init__(self, api_key=None, base_url=None, tempurature=0):
        self.api_key = api_key
        self.base_url = base_url
        self.tempurature = tempurature
        if api_key is None or base_url is None:
            logger.info("Loading environment variables from .env file")
            load_dotenv(find_dotenv())
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("BASE_URL")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        model="gpt-4o",
        stream=True,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    ):
        completion = self.client.chat.completions.create(
            model=model,
            stream=True,
            messages=messages,
        )
        return completion


model = largeModel()

model.chat()
