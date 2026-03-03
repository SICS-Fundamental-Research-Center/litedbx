import sys
from pathlib import Path

# Add parent directory to sys.path to allow importing from sibling directories
sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm
import instructor
from instructor.mode import Mode
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Optional, List, Dict, Any, Type, Tuple
from pydantic import BaseModel
from itertools import groupby
import base64
from dotenv import load_dotenv
import os
import yaml

load_dotenv()


class PromptParams:
    def __init__(self, kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs

    def setup_prompt(self, prompt: str) -> None:
        self.kwargs["messages"] = [{"role": "user", "content": []}]
        self.add_data_item(prompt, "Text")

    def add_data_item(self, data_item: str, modality: str) -> None:
        if modality == "Image":
            file_path = Path(data_item)
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
        elif modality == "Text":
            self.kwargs["messages"][-1]["content"].append(
                {"type": "text", "text": data_item}
            )
        else:
           raise NotImplementedError(f"Unsupported modality: {modality}")
         
    def structuring(self, response_model: Type[BaseModel]) -> None:
        self.kwargs["extra_body"] = {"enable_thinking": False}
        self.kwargs["response_model"] = response_model



class LdbLLMClient:
    def __init__(self):
        self.client = litellm
        self.client_struct_json = instructor.from_litellm(
            litellm.completion, mode=Mode.JSON)
        self.client_struct_async_json = instructor.from_litellm(
            litellm.acompletion, mode=Mode.JSON)
        self.client_struct_tools = instructor.from_litellm(
            litellm.completion, mode=Mode.TOOLS)
        self.client_struct_async_tools = instructor.from_litellm(
            litellm.acompletion, mode=Mode.TOOLS)

        with open(Path(__file__).parent / "config.yaml") as f:
            self.config = yaml.safe_load(f)

        self.max_retries = self.config.get("max_retries")
        self.parallelism = self.config.get("parallelism")
        self.sem = asyncio.Semaphore(self.parallelism)

        self.usage_statistics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "total_cost": 0.0,
        }

        self.kwargs = {
            "timeout": self.config.get("timeout"),
            "max_retries": self.config.get("max_retries"),
            "max_tokens": self.config.get("max_tokens"),
            "top_p": self.config.get("top_p"),
            "temperature": self.config.get("temperature"),
            "random_seed": self.config.get("random_seed"),
        }

    def get_usage_statistics(self):
        return self.usage_statistics

    def reset_usage_statistics(self):
        self.usage_statistics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "total_cost": 0.0,
        }

    def invoke(
            self,
            is_remote: bool,
            modality: str,
            prompt: str,
            data_items: Optional[List[str]] = None,
            response_model: Optional[Type[BaseModel]] = None,
            model_id: Optional[int] = None,
            enable_token_usage: bool = True,
    ):
        params, cost_params, mode = self._construct_prompt_params(
            is_remote=is_remote,
            modality=modality,
            prompt=prompt,
            data_items=data_items,
            response_model=response_model,
            model_index=model_id,
        )

        if response_model:
            return self._invoke_structured(params, enable_token_usage, cost_params, mode)
        else:
            return self._invoke(params, enable_token_usage, cost_params)


    async def invoke_parallel(
        self,
        is_remote: bool,
        modality: str,
        prompt: str,
        data_items: List[List[str]],
        response_model: Optional[Type[BaseModel]] = None,
        model_id: Optional[int] = None,
        enable_token_usage: bool = True,
    ):
        tasks = []
        for idx, data_item in enumerate(data_items):
            params, cost_params, mode = self._construct_prompt_params(
                is_remote=is_remote,
                modality=modality,
                prompt=prompt,
                data_items=data_item,
                response_model=response_model,
                model_index=model_id,
            )
            if response_model:
                tasks.append(self._ainvoke_structured(idx, params, enable_token_usage, cost_params, mode))
            else:
                tasks.append(self._ainvoke(idx, params, enable_token_usage, cost_params))

        results = await tqdm_asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        return [resp for _, resp in results]



    def _construct_prompt_params(
            self, 
            is_remote: bool,
            modality: str,
            prompt: str,
            data_items: Optional[List[str]] = None,
            response_model: Optional[Type[BaseModel]] = None,
            model_index: Optional[int] = None,
    ) -> Tuple[PromptParams, dict, str]:
        if not model_index:
            model_index = 0
        kwargs, cost_params, mode = \
            self._get_model_kw(is_remote=is_remote, modality=modality, model_index=model_index)
        params = PromptParams(kwargs=kwargs)
        params.setup_prompt(prompt)
        if data_items:
            for data_item in data_items:
                params.add_data_item(data_item, modality=modality)
        if response_model:
            params.structuring(response_model)
        return params, cost_params, mode


    def _get_model_kw(self, is_remote: bool, modality: str, model_index: Optional[int] = None):
        if not model_index:
            model_index = 0
        lm_type = "REMOTE_MODELS" if is_remote else "LOCAL_MODELS"

        lm_params = self.config.get(lm_type).get(modality)
        assert len(lm_params) > model_index, \
            f"No model found for type {lm_type} and modality {modality} at index {model_index}"

        cost_params = {
            "input_cost": lm_params[model_index].get("input_cost"),
            "output_cost": lm_params[model_index].get("output_cost"),
            "output_cost_thinking": lm_params[model_index].get("output_cost_thinking")
        }
        mode = lm_params[model_index].get("mode", "JSON").upper()

        if lm_type == "REMOTE_MODELS":
            lm_params = {
                "model": lm_params[model_index]["model"],
                "api_key": os.getenv(lm_params[model_index]["api_key"]),
                "api_base": os.getenv(lm_params[model_index]["api_base"]),
            }
        else:
            lm_params = {
                "model": lm_params[model_index]["model"],
                "api_key": "*",
                "api_base": lm_params[model_index]["api_base"],
            }

        return  dict(lm_params, **self.kwargs), cost_params, mode

    def _invoke(self, params: PromptParams, enable_token_usage: bool = True, 
                cost_params: Optional[dict] = None):
        resp =  self.client.completion(**params.kwargs)
        if enable_token_usage:
            assert cost_params is not None, "Cost parameters must be provided when token usage tracking is enabled."
            usage = self._extract_usage(resp)
            self.usage_statistics["prompt_tokens"] += usage["prompt_tokens"]
            self.usage_statistics["completion_tokens"] += usage["completion_tokens"]
            self.usage_statistics["total_tokens"] += usage["total_tokens"]
            self.usage_statistics["prompt_cost"] += \
                usage["prompt_tokens"] * cost_params["input_cost"] / 1000 / 1000
            self.usage_statistics["completion_cost"] += \
                usage["completion_tokens"] * cost_params["output_cost"] / 1000 / 1000
            self.usage_statistics["total_cost"] = \
                self.usage_statistics["prompt_cost"] + self.usage_statistics["completion_cost"]
        return resp

    def _invoke_structured(self, params: PromptParams, enable_token_usage: bool = True, 
                           cost_params: Optional[dict] = None, mode: str = "JSON"):
        _client = None
        if mode == "JSON":
            _client = self.client_struct_json
        elif mode == "TOOLS":
            _client = self.client_struct_tools
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        # Call instructor - mode is already set during initialization
        resp, completion = _client.create_with_completion(
            **params.kwargs
        )
        if enable_token_usage:
            assert cost_params is not None, "Cost parameters must be provided when token usage tracking is enabled."
            usage = self._extract_usage(completion)
            self.usage_statistics["prompt_tokens"] += usage["prompt_tokens"]
            self.usage_statistics["completion_tokens"] += usage["completion_tokens"]
            self.usage_statistics["total_tokens"] += usage["total_tokens"]
            self.usage_statistics["prompt_cost"] += \
                usage["prompt_tokens"] * cost_params["input_cost"] / 1000 / 1000
            self.usage_statistics["completion_cost"] += \
                usage["completion_tokens"] * cost_params["output_cost"] / 1000 / 1000
            self.usage_statistics["total_cost"] = \
                self.usage_statistics["prompt_cost"] + self.usage_statistics["completion_cost"]
        return resp

    async def _ainvoke(self, worker_id, params: PromptParams, enable_token_usage: bool = True, 
                       cost_params: Optional[dict] = None):
        attempt = 0

        while attempt <= self.max_retries:
            try:
                async with self.sem:
                    resp = await self.client.acompletion(**params.kwargs)
                    if enable_token_usage:
                        assert cost_params is not None, \
                            "Cost parameters must be provided when token usage tracking is enabled."
                        usage = self._extract_usage(resp)
                        self.usage_statistics["prompt_tokens"] += usage["prompt_tokens"]
                        self.usage_statistics["completion_tokens"] += usage["completion_tokens"]
                        self.usage_statistics["total_tokens"] += usage["total_tokens"]
                        self.usage_statistics["prompt_cost"] += \
                            usage["prompt_tokens"] * cost_params["input_cost"] / 1000 / 1000
                        self.usage_statistics["completion_cost"] += \
                            usage["completion_tokens"] * cost_params["output_cost"] / 1000 / 1000
                        self.usage_statistics["total_cost"] = \
                            self.usage_statistics["prompt_cost"] + self.usage_statistics["completion_cost"]
                    return (worker_id, resp)
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise e
                await asyncio.sleep(min(2**attempt, 30))

    async def _ainvoke_structured(self, worker_id, params: PromptParams, enable_token_usage: bool = True, 
                                  cost_params: Optional[dict] = None, mode: str = "JSON"):

        _client = None
        if mode == "JSON":
            _client = self.client_struct_async_json
        elif mode == "TOOLS":
            _client = self.client_struct_async_tools
        else:
            raise ValueError(f"Unsupported mode: {mode}")        


        attempt = 0

        while attempt <= self.max_retries:
            try:
                async with self.sem:
                    resp, completion = (
                        await _client.create_with_completion(
                            **params.kwargs
                        )
                    )
                    if enable_token_usage:
                        assert cost_params is not None, \
                            "Cost parameters must be provided when token usage tracking is enabled."
                        usage = self._extract_usage(completion)
                        self.usage_statistics["prompt_tokens"] += usage["prompt_tokens"]
                        self.usage_statistics["completion_tokens"] += usage["completion_tokens"]
                        self.usage_statistics["total_tokens"] += usage["total_tokens"]
                        self.usage_statistics["prompt_cost"] += \
                            usage["prompt_tokens"] * cost_params["input_cost"] / 1000 / 1000
                        self.usage_statistics["completion_cost"] += \
                            usage["completion_tokens"] * cost_params["output_cost"] / 1000 / 1000
                        self.usage_statistics["total_cost"] = \
                            self.usage_statistics["prompt_cost"] + self.usage_statistics["completion_cost"]
                    return (worker_id, resp)
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise e
                await asyncio.sleep(min(2**attempt, 30))
    
    def _extract_usage(self, resp):
        assert resp is not None, "Response is None, cannot extract token usage."
        usage = getattr(resp, "usage", None)
        assert usage is not None, "Token usage information is missing in the response."

        # normalize to a stable schema
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    def _test_invoke(self):
        from data_structure import BooleanFeatureResponse

        prompt = "Does this X-Ray indicate pneumonia? Answer with True or False."

        resp_remote = self.invoke(
            is_remote=True,
            modality="Image",
            prompt=prompt,
            data_items=["../files/medical/data/raw_data/all_x_rays/0_06_encapsulated_lesions_06 (204).jpeg"],
            response_model=BooleanFeatureResponse,
        )
        print(f"Remote response: {resp_remote}")

        resp_local = self.invoke(
            is_remote=False,
            modality="Image",
            prompt=prompt,
            data_items=["../files/medical/data/raw_data/all_x_rays/0_06_encapsulated_lesions_06 (204).jpeg"],
            response_model=BooleanFeatureResponse,
        )
        print(f"Local response: {resp_local}")

    async def _atest_invoke(self):
        from data_structure import BooleanFeatureResponse

        prompt = "Does this X-Ray indicate pneumonia? Answer with True or False."

        resp = await self.invoke_parallel(
            is_remote=True,
            modality="Image",
            prompt=prompt,
            data_items=[["../files/medical/data/raw_data/all_x_rays/0_06_encapsulated_lesions_06 (204).jpeg"]],
            response_model=BooleanFeatureResponse,
        )
        print(f"Local response: {resp}")


        
if __name__ == "__main__":
    llm_client = LdbLLMClient()

    llm_client._test_invoke()
    # asyncio.run(llm_client._atest_invoke())

    print(llm_client.usage_statistics)
