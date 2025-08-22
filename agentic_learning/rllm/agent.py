import copy

from rllm.agents.agent import BaseAgent, Trajectory, Step, Action


AGENTIC_LEARNING_PROMPT = """
**Objective:** 

To answer a User's question by providing a clear, verifiable reasoning process, potentially interacting with an external environment.

**Interaction Protocol:**

For each question you receive, you MUST follow this two-step process:

**Step 1: Reasoning Process**
*   **Decomposition:** Break down the user's question into a logical, step-by-step sequence of reasoning. Start from the most basic facts and build upon them.
*   **External Inquiry (Optional but Encouraged):**
    *   You may issue up to {max_turns} queries to an External Environment to validate hypotheses, clarify information, or advance your reasoning.
    *   Each query must be a self-contained question enclosed in `<query>...</query>` tags.
    *   **Wait for the `<result>...</result>` block** from the environment before continuing your reasoning.
    *   **Critically analyze and integrate** the content from the `<result>` into your reasoning chain.
    *   Do not invent, assume, or hallucinate any `<result>` content. Your reasoning must be grounded in the provided results.

**Step 2: Final Answer**
*   After your reasoning is complete, state your final answer clearly.
*   The final answer, and only the final answer, MUST be enclosed in `\\boxed{{...}}`.

**Output Format:**

Your final output must strictly adhere to the following structure. Do not add any preambles or postscripts.

```
**Reasoning:**  
(Your detailed, step-by-step reasoning, interleaving narrative, `<query>`, and `<result>` blocks as needed.)

**Answer:**  
\\boxed{{...your final answer...}}
```
""".strip()

EVAL_PROMPT = """
Please answer the question step by step and put your final answer (and only answer) within \\boxed{{}}
""".strip()


class AgenticLearningAgent(BaseAgent):

    def __init__(
        self, 
        enable_interaction: bool = True,
        max_turns: int = 5
    ):
        super().__init__()
        self.enable_interaction = enable_interaction
        self.max_turns = max_turns

        self._trajectory = Trajectory()
        self.train_system_prompt = [
            {
                "role": "system", 
                "content": AGENTIC_LEARNING_PROMPT.format(max_turns=self.max_turns) if self.enable_interaction else EVAL_PROMPT
            }
        ]
        self.val_system_prompt = [
            {"role": "system", "content": EVAL_PROMPT}
        ]
        self.prompt = None
        self.update_response = ""
        self.response = ""

        self.knowledge_messages = []
        self.response_tokens = []
        self.response_mask = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        if self.response == "":
            return self.prompt
        else:
            return [
                *self.prompt,
                {"role": "assistant", "content": self.response}
            ]

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation: dict, reward: float, done: bool, info: dict, **kwargs):
        if done:
            self.update_response = ""
            return
    
        if self.prompt is None:
            # Initial Task
            assert isinstance(observation, dict) and "question" in observation
            assert isinstance(observation, dict) and "split" in observation
            question = observation["question"]
            split = observation["split"]
            self.prompt = [
                self.train_system_prompt[0] if split == "train" else self.val_system_prompt[0],  
                {"role": "user", "content": question}
            ]
        else:
            assert isinstance(observation, dict) and "query" in observation
            assert isinstance(observation, dict) and "result" in observation
            if "content" in observation:
                self.knowledge_messages.append(
                    [
                        {"role": "user", "content": observation["query"]},
                        {"role": "assistant", "content": observation["content"]}
                    ]
                )
            self.update_response = f"<result>{observation['result']}</result>"
            self.response += self.update_response

            tokens = kwargs["tokenizer"].encode(self.update_response, add_special_tokens=False)
            self.response_tokens.extend(tokens)
            self.response_mask.extend([0] * len(tokens))

    def update_from_model(self, response: str, **kwargs) -> Action:
        self.update_response = response
        self.response += self.update_response

        tokens = kwargs["tokenizer"].encode(self.update_response, add_special_tokens=False)
        self.response_tokens.extend(tokens)
        self.response_mask.extend([1] * len(tokens))

        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions))
        self.trajectory.steps.append(new_step)

        return Action(action=response)

    def reset(self):
        self._trajectory = Trajectory()
        self.train_system_prompt = [
            {
                "role": "system", 
                "content": AGENTIC_LEARNING_PROMPT.format(max_turns=self.max_turns) if self.enable_interaction else EVAL_PROMPT
            }
        ]
        self.val_system_prompt = [
            {"role": "system", "content": EVAL_PROMPT}
        ]
        self.prompt = None
        self.update_response = ""
        self.response = ""

        self.knowledge_messages = []
        self.response_tokens = []
        self.response_mask = []

    def get_current_state(self) -> Step | None:
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
