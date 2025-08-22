import openai
import re
import traceback

from rllm.environments.base.base_env import BaseEnv
from rllm.misc import colorful_print

from agentic_learning.verl.reward import last_boxed_only_string


class AgenticLearningEnv(BaseEnv):

    def __init__(
        self, 
        task: dict | None = None, 
        max_turns: int = 5,
        model_name: str = "gpt-4o-mini",
        base_url: str | list[str] = "https://api.openai.com/v1",
        api_key: str = "EMPTY",
        max_retries: int = 3
    ):
        super().__init__()
        self.task = task
        self.max_turns = max_turns
        self.model_name = model_name
        self.max_retries = max_retries

        if isinstance(base_url, str):
            base_url = [base_url]

        self.clients = [
            openai.OpenAI(base_url=url, api_key=api_key) 
            for url in base_url
        ]

        self.done = False
        self.current_turn = 0

    def reset(self, task=None, seed=None):
        if task is not None:
            self.task = task

        self.done = False
        self.current_turn = 0

        assert self.task is not None, "Task must be set before reset"

        return self.task, {}
    
    def step(self, action: str):
        action = action.strip()
        
        if not action.endswith("</query>"):
            self.done = True
            return (
                {},
                0,
                True,
                self.task
            )

        query = re.search(r"<query>(.*?)</query>", action)
        if query:
            query = query.group(1)
        else:
            query = ""

        if query == "":
            return (
                {
                    "query": "",
                    "result": "Error: No question found maybe due to the incorrect format."
                }, 
                0, 
                False, 
                self.task
            )

        if self.current_turn >= self.max_turns:
            return (
                {
                    "query": "",
                    "result": "Error: Reached the maximum number of questions."
                }, 
                0, 
                False, 
                self.task
            )
        
        num_tries = 0
        colorful_print(f"deal with query: {query}\n", "blue")
        while num_tries < self.max_retries:
            try:
                completion = self.clients[self.idx % len(self.clients)].chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user", "content": query + "\n\nPut your answer within \\boxed{{}}."
                        }
                    ],
                    max_tokens=1024*16,
                    n=1,
                    temperature=0.7,
                    top_p=0.8
                )

                self.current_turn += 1

                result = last_boxed_only_string(completion.choices[0].message.content) 

                if result:
                    return (
                        {
                            "query": query,
                            "result": result,
                            "content": completion.choices[0].message.content,
                            "overlong": completion.choices[0].finish_reason == "length"
                        }, 
                        1, 
                        False, 
                        self.task
                    )
                else:
                    return (
                        {
                            "query": query,
                            "result": completion.choices[0].message.content[-200:],
                            "content": completion.choices[0].message.content,
                        }, 
                        0, 
                        False, 
                        self.task
                    )
            except Exception as e:
                colorful_print(traceback.format_exc())
                num_tries += 1

        self.current_turn += 1

        return (
            {
                "query": "",
                "result": f"Error: {traceback.format_exc()}."
            }, 
            0, 
            False, 
            self.task
        )

    @staticmethod
    def from_dict(env_args: dict) -> "AgenticLearningEnv":
        return AgenticLearningEnv(
            task={
                "question": env_args.get("question"),
                "split": env_args.get("split")
            },
            max_turns=env_args.get("max_turns", 5),
            base_url=env_args.get("base_url", "https://api.openai.com/v1"),
            api_key=env_args.get("api_key", ""),
            max_retries=env_args.get("max_retries", 3)
        )