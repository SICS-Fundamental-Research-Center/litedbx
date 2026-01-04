import litellm
import instructor
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Optional, List, Dict, Any, Type
from pydantic import BaseModel
from pathlib import Path
import base64
from dotenv import load_dotenv
import os

load_dotenv()


class PromptParams:
    def __init__(self, kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs

    def setup_prompt(self, prompt: str) -> None:
        self.kwargs["messages"] = [{"role": "user", "content": []}]
        self.add_data_item(prompt)

    def add_data_item(self, data_item: str) -> None:
        file_path = Path(data_item)
        if any(
            data_item.endswith(extension) for extension in [".png", ".jpg", ".jpeg"]
        ):
            with file_path.open("rb") as image_file:
                image = base64.b64encode(image_file.read()).decode("utf-8")
            self.kwargs["messages"][-1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                        "detail": "low",
                    },
                }
            )
        elif any(data_item.endswith(extension) for extension in [".wav", ".mp3"]):
            raise NotImplementedError("Audio data encoding is not implemented yet.")
        elif any(
            data_item.endswith(extension) for extension in [".mp4", ".avi", ".mov"]
        ):
            raise NotImplementedError("Video data encoding is not implemented yet.")
        else:
            self.kwargs["messages"][-1]["content"].append(
                {"type": "text", "text": data_item}
            ) 

    def structuring(self, response_model: Type[BaseModel]) -> None:
        self.kwargs["extra_body"] = {"enable_thinking": False}
        self.kwargs["response_model"] = response_model



class LiteLLMWrapper:
    def __init__(self):
        self.client = litellm
        self.client_struct = instructor.from_litellm(litellm.completion)
        self.client_struct_async = instructor.from_litellm(litellm.acompletion)

        self.max_retries = 50
        self.parallelism = 100
        self.sem = asyncio.Semaphore(self.parallelism)

        self.kwargs = {
            "timeout": 30,
            "max_retries": 3,
            "max_tokens": 4096,
            "top_p": 1.0,
            "temperature": 0.0,
            "random_seed": 42,
        }

        self.remote_text_kwargs = {
            "model": "openai/Qwen3-235B-A22B",
            "api_key": os.getenv("BLSC_API_KEY"),
            "api_base": os.getenv("BLSC_ENDPOINT"),
        }
        self.remote_vision_kwargs = {
            "model": "openai/Qwen3-VL-235B-A22B-Thinking",
            "api_key": os.getenv("BLSC_API_KEY"),
            "api_base": os.getenv("BLSC_ENDPOINT"),
        }
        self.local_text_kwargs = {
            "model": "hosted_vllm//ssd_data/models/llama3-8b-instruct/",
            "api_key": "*",
            "api_base": "http://localhost:8000/v1",
        }

    def invoke(
            self, 
            is_remote: bool, 
            modality: str, 
            prompt: str, 
            data_items: Optional[str] = None,
            reponse_model: Optional[Type[BaseModel]] = None
    ):
        params = PromptParams(
            kwargs=self._get_model_kw(is_remote, modality)
        )
        params.setup_prompt(prompt)
        if data_items:
            params.add_data_item(data_items)
        if reponse_model:
            params.structuring(reponse_model)
            return self.client_struct.chat.completions.create(**params.kwargs)
        else:
            return self.client.completion(**params.kwargs)

    async def invoke_parallel(
        self,
        is_remote: bool,
        modality: str,
        prompt: str,
        data_items: List[str],
        response_model: Optional[Type[BaseModel]] = None,
    ):
        async def exec(worker_id: int, prompt_params: PromptParams):
            attempt = 0

            while attempt <= self.max_retries:
                try:
                    async with self.sem:
                        if response_model is not None:
                            resp = (
                                await self.client_struct_async.chat.completions.create(
                                    **prompt_params.kwargs
                                )
                            )
                        else:
                            resp = await self.client.acompletion(**prompt_params.kwargs)
                        return (worker_id, resp)
                except Exception as e:
                    attempt += 1

                    if attempt > self.max_retries:
                        raise e

                    await asyncio.sleep(min(2**attempt, 30))

        tasks = []
        for idx, data_item in enumerate(data_items):
            params = PromptParams(
                kwargs=self._get_model_kw(is_remote=is_remote, modality=modality)
            )
            params.setup_prompt(prompt)
            params.add_data_item(data_item)
            if response_model is not None:
                params.structuring(response_model)

            tasks.append(exec(idx, params))

        results = await tqdm_asyncio.gather(*tasks)

        results.sort(key=lambda x: x[0])
        return [resp for _, resp in results]
    
    
    def _get_model_kw(self, is_remote: bool, modality: str):
        if is_remote and modality == "TEXT":
            return dict(self.remote_text_kwargs, **self.kwargs)
        elif is_remote and modality == "VISION":
            return dict(self.remote_vision_kwargs, **self.kwargs)
        elif not is_remote and modality == "TEXT":
            return dict(self.local_text_kwargs, **self.kwargs)
        else:
            raise ValueError(
                f"Invalid model config for is_remote={is_remote}, modality={modality}")




if __name__ == "__main__":
    llm_client = LiteLLMWrapper()
    print(llm_client._get_model_kw(is_remote=False, modality="TEXT"))
    result = llm_client.invoke(
        is_remote=False,
        modality="TEXT",
        prompt="Hello, how are you?",
    )
    print("=" * 50)
    print(result)
    print("=" * 50)
