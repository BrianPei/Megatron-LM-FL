# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pydantic import PrivateAttr

from megatron.rl.agent.reward_only_agent import RewardOnlyAgent


class NullNumericAgent(RewardOnlyAgent):
    """Minimal self-contained RL agent for functional testing.

    Prompts are numeric token strings so they work with NullTokenizer without
    any external tokenizer assets or datasets.
    """

    env_id: str = "null_numeric"
    prompt_pool_size: int = 16
    prompt_length: int = 4
    validation_size: int = 8

    _train_data: list[dict] = PrivateAttr(default_factory=list)
    _valid_data: list[dict] = PrivateAttr(default_factory=list)
    _next_index: int = PrivateAttr(default=0)

    def model_post_init(self, __context) -> None:
        self._train_data = [self._build_entry(i) for i in range(self.prompt_pool_size)]
        self._valid_data = [self._build_entry(100 + i) for i in range(self.validation_size)]

    def _build_entry(self, seed: int) -> dict:
        prompt_tokens = [((seed + offset) % 120) + 1 for offset in range(self.prompt_length)]
        return {
            "problem_id": f"null_numeric_{seed}",
            "prompt_text": " ".join(str(token) for token in prompt_tokens),
        }

    def get_dataset(self, validation: bool = False):
        return self._valid_data if validation else self._train_data

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> list[tuple[str, dict]]:
        dataset = self.get_dataset(validation)
        return [(entry["prompt_text"], entry) for entry in dataset[:num_prompts]]

    async def get_prompt(self, validation: bool = False) -> tuple[str, dict]:
        dataset = self.get_dataset(validation)
        entry = dataset[self._next_index % len(dataset)]
        self._next_index += 1
        return entry["prompt_text"], entry

    async def get_reward(self, response: str, golden: dict) -> float:
        tokens = [int(tok) for tok in response.split() if tok.lstrip("-").isdigit()]
        if not tokens:
            return 0.0
        return 1.0 if tokens[0] % 2 == 0 else 0.0
