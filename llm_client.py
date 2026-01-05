import litellm
import instructor
from instructor.mode import Mode
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Optional, List, Dict, Any, Type
from pydantic import BaseModel
from pathlib import Path
from itertools import groupby
import base64
from dotenv import load_dotenv
import os

load_dotenv()

class BooleanFeatureResponse(BaseModel):
    """Response model for boolean feature extraction."""
    value: bool


class NumericalFeatureResponse(BaseModel):
    """Response model for numerical feature extraction."""
    value: float


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
        # Use Mode.JSON when initializing instructor to avoid function calling with vLLM
        self.client_struct = instructor.from_litellm(litellm.completion, mode=Mode.JSON)
        self.client_struct_async = instructor.from_litellm(litellm.acompletion, mode=Mode.JSON)

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

        self.lm_kwargs = {
            "REMOTE": {
                "TEXT": [
                    {
                        "model": "openai/Qwen3-235B-A22B",
                        "api_key": os.getenv("BLSC_API_KEY"),
                        "api_base": os.getenv("BLSC_ENDPOINT"),
                    },
                ],
                "IMAGE": [
                    {
                        "model": "openai/Qwen3-VL-235B-A22B-Instruct",
                        "api_key": os.getenv("BLSC_API_KEY"),
                        "api_base": os.getenv("BLSC_ENDPOINT"),
                    },
                ]
            },
            "LOCAL": {
                "TEXT": [
                    {
                        "model": "hosted_vllm//ssd_data/models/llama3-8b-instruct/",
                        "api_key": "*",
                        "api_base": "http://localhost:8000/v1",
                    },
                    {
                        "model": "hosted_vllm//ssd_data/models/Qwen3-4B-Instruct-2507",
                        "api_key": "*",
                        "api_base": "http://localhost:8001/v1",
                    }
                ],
                "IMAGE": [
                    {
                        "model": "hosted_vllm//ssd_data/models/Qwen3-VL-8B-Instruct",
                        "api_key": "*",
                        "api_base": "http://localhost:8002/v1",
                    },
                    {
                        "model": "hosted_vllm//ssd_data/models/llava-v1___6-mistral-7b-hf",
                        "api_key": "*",
                        "api_base": "http://localhost:8003/v1",
                    }
                ]
            }
        }

    def invoke(
            self, 
            is_remote: bool, 
            modality: str, 
            prompt: str, 
            data_item: Optional[str] = None,
            response_model: Optional[Type[BaseModel]] = None,
            model_id: Optional[int] = None,
    ):
        params = self._construct_prompt_params(
            is_remote=is_remote,
            modality=modality,
            prompt=prompt,
            data_item=data_item,
            response_model=response_model,
            model_index=model_id,
        )

        if response_model:
            return self._invoke_structured(params)
        else:
            return self._invoke(params)


    def invoke_with_proxy(
            self,
            modality: str,
            prompt: str,
            data_item: Optional[str] = None,
            response_model: Optional[Type[BaseModel]] = None,
    ):
        assert len(self.lm_kwargs.get("LOCAL", {}).get(modality, [])) >= 2, \
            f"At least two local {modality} models are required for proxy chat."
        
        results = set()
        for model_id in range(len(self.lm_kwargs.get("LOCAL", {}).get(modality, []))):
            params = self._construct_prompt_params(
                is_remote=False,
                modality=modality,
                prompt=prompt,
                data_item=data_item,
                response_model=response_model,
                model_index=model_id,
            )
            if response_model:
                result = self._invoke_structured(params=params)
            else:
                result = self._invoke(params=params)
            results.add(result)

        if len(results) == 1:
            print("Consensus reached among local models. The result is accepted.")
            return results.pop()
        else:
            print("No consensus among local models. Invoking remote model.")
            return self.invoke(
                is_remote=True,
                modality=modality,
                prompt=prompt,
                data_item=data_item,
                response_model=response_model,
            )

    async def invoke_parallel(
        self,
        is_remote: bool,
        modality: str,
        prompt: str,
        data_items: List[str],
        response_model: Optional[Type[BaseModel]] = None,
        model_id: Optional[int] = None,
    ):
        tasks = []
        for idx, data_item in enumerate(data_items):
            params = self._construct_prompt_params(
                is_remote=is_remote,
                modality=modality,
                prompt=prompt,
                data_item=data_item,
                response_model=response_model,
                model_index=model_id,
            )
            if response_model:
                tasks.append(self._ainvoke_structured(idx, params))
            else:
                tasks.append(self._ainvoke(idx, params))

        results = await tqdm_asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        return [resp for _, resp in results]

    async def invoke_parallel_with_proxy(
        self,
        modality: str,
        prompt: str,
        data_items: List[str],
        response_model: Optional[Type[BaseModel]] = None,
    ):
        assert len(self.lm_kwargs.get("LOCAL", {}).get(modality, [])) >= 2, \
            f"At least two local {modality} models are required for proxy chat."
        
        tasks = []
        for idx, data_item in enumerate(data_items):
            for model_id in range(len(self.lm_kwargs.get("LOCAL", {}).get(modality, []))):
                params = self._construct_prompt_params(
                    is_remote=False,
                    modality=modality,
                    prompt=prompt,
                    data_item=data_item,
                    response_model=response_model,
                    model_index=model_id,
                )
                if response_model:
                    tasks.append(self._ainvoke_structured(idx, params))
                else:
                    tasks.append(self._ainvoke(idx, params))
        results = await tqdm_asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])

        # Filter results by consensus
        sound_results, filtered_tasks = [], []
        result_groups = [list(group) for key, group in groupby(results, key=lambda x: x[0])]
        for group in result_groups:
            assert len(group) == len(self.lm_kwargs.get("LOCAL", {}).get(modality, [])), \
                f"Each data item should have results from all local {modality} models."
            resp_set = set([resp for _, resp in group])
            if len(resp_set) == 1:
                sound_results.append(group[0])
            else:
                # No consensus, need to invoke remote model
                params = self._construct_prompt_params(
                    is_remote=True,
                    modality=modality,
                    prompt=prompt,
                    data_item=data_items[group[0][0]],
                    response_model=response_model,
                )
                if response_model:
                    filtered_tasks.append(self._ainvoke_structured(group[0][0], params))
                else:
                    filtered_tasks.append(self._ainvoke(group[0][0], params))
        print(f"# Consensus results: {len(sound_results)}, # Recomputing tasks: {len(filtered_tasks)}")
        recomputed_results = await tqdm_asyncio.gather(*filtered_tasks)
        recomputed_results.sort(key=lambda x: x[0])
        proxy_mask = [False for _ in range(len(data_items))]  # Track which items used proxy
        for idx, _ in sound_results:
            proxy_mask[idx] = True
        sound_results.extend(recomputed_results)
        sound_results.sort(key=lambda x: x[0])
        llm_labels = [resp for _, resp in sound_results]
        return proxy_mask, llm_labels


    def _construct_prompt_params(
            self, 
            is_remote: bool,
            modality: str,
            prompt: str,
            data_item: Optional[str] = None,
            response_model: Optional[Type[BaseModel]] = None,
            model_index: Optional[int] = None,
    ) -> PromptParams:
        if not model_index:
            model_index = 0
        params = PromptParams(
            kwargs=self._get_model_kw(is_remote=is_remote, modality=modality, model_index=model_index)
        )
        params.setup_prompt(prompt)
        if data_item:
            params.add_data_item(data_item)
        if response_model:
            params.structuring(response_model)
        return params
    
    def _invoke(self, params: PromptParams):
        resp =  self.client.completion(**params.kwargs)
        return resp

    def _invoke_structured(self, params: PromptParams):
        # Call instructor - mode is already set during initialization
        resp = self.client_struct.chat.completions.create(
            **params.kwargs
        )
        return resp.value

    async def _ainvoke(self, worker_id, params: PromptParams):
        attempt = 0

        while attempt <= self.max_retries:
            try:
                async with self.sem:
                    resp = await self.client.acompletion(**params.kwargs)
                    return (worker_id, resp)
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise e
                await asyncio.sleep(min(2**attempt, 30))


    async def _ainvoke_structured(self, worker_id, params: PromptParams):
        attempt = 0

        while attempt <= self.max_retries:
            try:
                async with self.sem:
                    # Call instructor - mode is already set during initialization
                    resp = (
                        await self.client_struct_async.chat.completions.create(
                            **params.kwargs
                        )
                    )
                    return (worker_id, resp.value)
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise e
                await asyncio.sleep(min(2**attempt, 30))
    
    
    def _get_model_kw(self, is_remote: bool, modality: str, model_index: Optional[int] = None):
        if not model_index:
            model_index = 0
        lm_type = "REMOTE" if is_remote else "LOCAL"

        lm_params = self.lm_kwargs.get(lm_type, {})\
                                  .get(modality, [])
        assert len(lm_params) > model_index, \
            f"No model found for type {lm_type} and modality {modality} at index {model_index}"

        return  dict(lm_params[model_index], **self.kwargs)


        
if __name__ == "__main__":
    llm_client = LiteLLMWrapper()


    result = llm_client.invoke_with_proxy(
        modality="TEXT",
        prompt="Analyze whether this patient has an allergy based on their symptoms. You must ONLY respond with a bool object (True if allergy is present, False otherwise).",
        data_item="Along with recurrent headaches and blurred vision, I suffer acid reflux and trouble digesting my food.",
        response_model=BooleanFeatureResponse,
    )

    print(result)

