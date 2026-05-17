import asyncio
import os
import time
from typing import List, Dict, Any, Callable, Optional
import numpy as np
import aiohttp

import ssl

class BestOfNSampler:

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "deepseek/deepseek-r1:free",
            n_samples: int = 5,
            temperature: float = 1.0,
            max_tokens: int = 1000,
            scoring_function: Optional[Callable] = None,
            timeout: int = 120,
            concurrent: bool = True
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set the OPENROUTER_API_KEY environment variable.")

        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.concurrent = concurrent
        self.api_base = "https://openrouter.ai/api/v1/chat/completions"

        self.scoring_function = scoring_function


    async def _generate_single(self, prompt: str, system_prompt: str, LLM_judge, temp: float) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",  # Required by OpenRouter
        }

        payload = {
            "model": LLM_judge,
            "max_tokens": self.max_tokens,
            "temperature": temp,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:

            async with session.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"API call failed with status {response.status}: {error_text}")

                response_data = await response.json()

                try:
                    return response_data['choices'][0]['message']['content']
                except (KeyError, IndexError) as e:
                    print("ERRR", e)
                    raise RuntimeError(f"Unexpected API response format: {e}")

    async def _generate_samples(self, prompt, system_prompt: str, LLM_judge, n_samples=0, temperature=0) -> List[str]:

        if n_samples == 0:
            n_samples = self.n_samples
        if temperature == 0:
            temperature = self.temperature
        if self.concurrent:
            tasks = []
            for i in range(n_samples):
                temp = temperature
                task = asyncio.create_task(
                    self._generate_single(prompt, system_prompt, LLM_judge, temp)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            samples = [r for r in results if not isinstance(r, Exception)]
            if len(samples) == 0:
                raise RuntimeError("All API calls failed")

            return samples
        else:
            samples = []
            for i in range(n_samples):
                temp = self.temperature
                try:
                    sample = await self._generate_single(prompt, system_prompt, LLM_judge, temp)
                    samples.append(sample)
                except Exception as e:
                    print(f"Error generating sample {i}: {e}")

            if len(samples) == 0:
                raise RuntimeError("All API calls failed")

            return samples

    def post_process(self, action):
        action = action.lower()
        action = action.replace(':', '')
        action = action.replace('action', '')
        action = action.replace('unload', '*unload')
        action = action.replace('deliver', '*deliver')
        action = action.replace('livingroom', 'living room')
        action = action.replace('drive', '*drive')
        if 'load' in action and 'unload' not in action:
            action = action.replace('load', '*load')

        if "drive" in action:
            action = action.replace('to ', 'to-')

        action = action.strip()
        return action

    async def generate(self, prompt: str, system_prompt: str = "", monitor=None, LLM_judge=None, n_samples=3, temperature=0.8, sample0=None) -> Dict[str, Any]:
        start_time = time.time()

        samples = await self._generate_samples(prompt, system_prompt, LLM_judge, n_samples, temperature)

        if sample0 is not None:
            samples.append(sample0)
            samples[-1] = samples[0]
            samples[0] = sample0

        for k in range(len(samples)):
            samples[k] = self.post_process(samples[k])


        scores = self.scoring_function(samples, monitor.__deepcopy__())

        sums_scores = np.array([np.sum(lst) for lst in scores])

        best_idx = np.argmax(sums_scores)
        best_sample = samples[best_idx]
        best_score = scores[best_idx]

        generation_time = time.time() - start_time

        return {
            "best_response": best_sample,
            "best_score": best_score,
            "best_idx": best_idx,
            "all_samples": samples,
            "all_scores": scores,
            "generation_time": generation_time,
            "num_samples_generated": len(samples)
        }


def trace_scorer(samples: List[str], source_monitor, manual=True):
    scores = []


    for sample in samples:
        monitor = source_monitor.__deepcopy__()
        t_scores = [0 for _ in range(len(monitor.current_formulas))]
        if manual:
            text = sample.lower()
            search_str = "**plan**"
            start_index = text.find(search_str)

            if start_index == -1:
                search_str = "plan"
                start_index = text.find(search_str)
                if start_index == -1:
                    start_index = 0
                else:
                    start_index = start_index + len(search_str)
            else:
                start_index = start_index + len(search_str)

            sentences = text[start_index:].split('\n')

            for s in sentences:
                if len(s) <= 3:
                    continue
                action = s.strip()
                monitor.step(action, "", 0)
                vio_which = monitor.history[-1]['vio-which']
                rules = vio_which.split('-')
                for r in rules:
                    if r.isdigit():
                        idx = int(r)
                        t_scores[idx - 1] += -1
            scores.append(t_scores)


    return scores
