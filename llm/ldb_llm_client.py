import sys
from pathlib import Path

# Add parent directory to sys.path to allow importing from sibling directories
sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm
litellm.telemetry = False
import instructor
from instructor.mode import Mode
from instructor.core.hooks import HookName, Hooks
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Optional, List, Dict, Any, Type, Tuple
from pydantic import BaseModel
import base64
from dotenv import load_dotenv
import os
import yaml
from PIL import Image
import io

load_dotenv()


class PromptParams:
    def __init__(self, kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs

    def setup_prompt(self, prompt: str) -> None:
        self.kwargs["messages"] = [{"role": "user", "content": []}]
        self.add_data_item(prompt, "Text")

    def add_data_item(self, data_item: str, modality: str, metadata: Optional[dict]=None) -> None:

        if metadata is not None:
            serialized_metadata = \
                "".join(f"{key.capitalize()}: {value}" for key, value in metadata.items())
            self.kwargs["messages"][-1]["content"].append(
                {"type": "text", "text": serialized_metadata}
            )

        if modality == "Image":
            file_path = Path(data_item)
            image = _encode_image_adaptive(file_path)
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



def _clean_tool_args(response: Any) -> Any:
    """Clean tool call arguments by stripping whitespace and fixing JSON issues."""
    if hasattr(response, 'choices') and response.choices:
        for choice in response.choices:
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                        # Strip leading/trailing whitespace from arguments
                        args = tool_call.function.arguments.strip()
                        # Fix Python-style boolean values (True/False) to JSON-style (true/false)
                        args = args.replace('True', 'true').replace('False', 'false')
                        tool_call.function.arguments = args
    return response




def _encode_image_adaptive(file_path: Path):
    MAX_PIXELS = 448 * 448      # threshold for resizing
    MAX_DIM = 640               # max width/height after resize
    JPEG_QUALITY = 85           # high quality to avoid degradation

    with Image.open(file_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        pixels = w * h

        # ---- Case 1: small image -> send original ----
        if pixels <= MAX_PIXELS:
            with file_path.open("rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # ---- Case 2: large image -> resize ----
        img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class LdbLLMClient:
    def __init__(self):
        self.client = litellm

        # Create hooks to clean tool call arguments
        hooks = Hooks()
        hooks.on(HookName.COMPLETION_RESPONSE, _clean_tool_args)

        self.client_struct_json = instructor.from_litellm(
            litellm.completion, mode=Mode.JSON)
        self.client_struct_async_json = instructor.from_litellm(
            litellm.acompletion, mode=Mode.JSON)
        self.client_struct_tools = instructor.from_litellm(
            litellm.completion, mode=Mode.TOOLS, hooks=hooks)
        self.client_struct_async_tools = instructor.from_litellm(
            litellm.acompletion, mode=Mode.TOOLS, hooks=hooks)

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
            data_items_metadata: Optional[List[dict]] = None,
            response_model: Optional[Type[BaseModel]] = None,
            model_id: Optional[int] = None,
            enable_token_usage: bool = True,
    ):
        invoke_modality = modality
        if modality == "VectorText":
            invoke_modality = "Text"
        if modality == "VectorImage":
            invoke_modality = "Image"

        if data_items and data_items_metadata and modality in ["VectorText", "VectorImage"]:
            flattened_data_items = []
            flattened_metadata = []
            for item, meta in zip(data_items, data_items_metadata):
                split_items = [i.strip() for i in item.split(",") if i.strip()][:5]
                flattened_data_items.extend(split_items)
                flattened_metadata.extend([meta] * len(split_items))
            data_items = flattened_data_items
            data_items_metadata = flattened_metadata

        params, cost_params, mode = self._construct_prompt_params(
            is_remote=is_remote,
            modality= invoke_modality,
            prompt=prompt,
            data_items=data_items,
            data_items_metadata=data_items_metadata,
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
        # Map phase: Expand data_items based on modality
        expanded_items = []
        original_indices = []  # Track which original item each expanded item belongs to

        for original_idx, data_item in enumerate(data_items):
            if modality == "VectorText" or modality == "VectorImage":
                # Split Vector content into individual items
                split_items = [item.strip() for item in data_item[0].split(",") if item.strip()]
                for split_item in split_items:
                    expanded_items.append([split_item])
                    original_indices.append(original_idx)
            else:
                # For non-Vector content, keep as-is
                expanded_items.append(data_item)
                original_indices.append(original_idx)

        # Create tasks for all expanded items
        tasks = []
        for exp_idx, exp_data_item in enumerate(expanded_items):
            invoke_modality = modality
            if modality == "VectorText":
                invoke_modality = "Text"
            if modality == "VectorImage":
                invoke_modality = "Image"
            params, cost_params, mode = self._construct_prompt_params(
                is_remote=is_remote,
                modality= invoke_modality,
                prompt=prompt,
                data_items=exp_data_item,
                response_model=response_model,
                model_index=model_id,
            )
            if response_model:
                tasks.append(self._ainvoke_structured(exp_idx, params, enable_token_usage, cost_params, mode))
            else:
                tasks.append(self._ainvoke(exp_idx, params, enable_token_usage, cost_params))

        # Execute all tasks in parallel
        results = await tqdm_asyncio.gather(*tasks)

        # Reduce phase: Group results by original indices and apply reduction logic
        if modality == "VectorText" or modality == "VectorImage":
            # Group results by original index
            grouped_results = {}
            for (exp_idx, resp), orig_idx in zip(results, original_indices):
                if orig_idx not in grouped_results:
                    grouped_results[orig_idx] = []
                grouped_results[orig_idx].append(resp)

            # Apply OR logic to each group
            final_results = []
            for orig_idx in sorted(grouped_results.keys()):
                reduced = self._reduce_vector_text_results(grouped_results[orig_idx])
                final_results.append((orig_idx, reduced))

            final_results.sort(key=lambda x: x[0])
            return [resp for _, resp in final_results]
        else:
            # For non-Vector content, return results as-is
            results.sort(key=lambda x: x[0])
            return [resp for _, resp in results]


    def _reduce_vector_text_results(self, results: list):
        if not results:
            return None

        from data_structure.llm_resp_templates import (
            BooleanFeatureResponse,
            IntFeatureResponse,
            FloatFeatureResponse,
        )

        # Get the type name of the first result
        result_type_name = type(results[0]).__name__

        # Check if all results have the same type.
        if not all(type(r).__name__ == result_type_name for r in results):
            raise ValueError((
                f"All results must have the same type, "
                f"got mixed types: {[type(r).__name__ for r in results]}"))

        # Handle BooleanFeatureResponse - apply OR logic
        if result_type_name == "BooleanFeatureResponse":
            reduced_value = any(r.value for r in results)  # OR logic on the value field
            return BooleanFeatureResponse(value=reduced_value)

        # Fallback for numerical features - apply SUM logic
        if result_type_name == "IntFeatureResponse":
            reduced_value = sum(r.value for r in results)
            return IntFeatureResponse(value=reduced_value)
        if result_type_name == "FloatFeatureResponse":
            reduced_value = sum(r.value for r in results)
            return FloatFeatureResponse(value=reduced_value)
            
        raise TypeError(f"Vector reduction is not supported for type {result_type_name}. "
                       f"Only BooleanFeatureResponse is supported.")



    def _construct_prompt_params(
            self,
            is_remote: bool,
            modality: str,
            prompt: str,
            data_items: Optional[List[str]] = None,
            data_items_metadata: Optional[list[dict]] = None,
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
            metadata =  data_items_metadata \
                if data_items_metadata is not None \
                    else [None] * len(data_items)
            for data_item, data_item_metadata in zip(data_items, metadata):
                params.add_data_item(data_item, modality=modality, metadata=data_item_metadata)
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
        from time import time

        # prompt = "Does this X-Ray indicate pneumonia? Answer with True or False."
        # image_ids = [
        #     204, 241, 529, 105, 591,
        #     140, 59, 628, 319, 471,
        #     434, 361, 324, 409, 138,
        #     64, 21, 615, 281, 239,
        # ]
        # start = time()
        # data_items = []
        # for idx, image_id in enumerate(image_ids):
        #     data_items.append([f"../files/medical/data/raw_data/all_x_rays/{idx}_06_encapsulated_lesions_06 ({image_id}).jpeg"])
        # resp = await self.invoke_parallel(
        #     is_remote=False,
        #     modality="Image",
        #     prompt=prompt,
        #     data_items=data_items,
        #     response_model=BooleanFeatureResponse,
        # )
        # end = time()
        # print(f"Parallel execution time for {len(image_ids)} images: {end - start} seconds")
        # print(f"Execution results: {resp}")


        prompt = "Does the displayed product show a (pair of) sports shoe(s) and the shoe(s) have the colors yellow and silver? Please answer with True or False."
        image_ids = [
            1163, 1164, 1165, 1525, 1526,
            1528, 1529, 1530, 1531, 1532,
            1533, 1534, 1535, 1536, 1537,
            1538, 1539, 1540, 1541, 1542,
        ]
        start = time()
        data_items = []
        for idx, image_id in enumerate(image_ids):
            data_items.append([f"../files/ecomm/source_data/fashion-dataset/images/{image_id}.jpg"])
        resp = await self.invoke_parallel(
            is_remote=False,
            modality="Image",
            prompt=prompt,
            data_items=data_items,
            response_model=BooleanFeatureResponse,
        )
        end = time()
        print(f"Parallel execution time for {len(image_ids)} images: {end - start} seconds")
        print(f"Execution results: {resp}")

        # resp = await self.invoke_parallel(
        #     is_remote=True,
        #     modality="Image",
        #     prompt=prompt,
        #     data_items=[["../files/medical/data/raw_data/all_x_rays/0_06_encapsulated_lesions_06 (204).jpeg"]],
        #     response_model=BooleanFeatureResponse,
        # )
        # print(f"Local response: {resp}")



if __name__ == "__main__":
    llm_client = LdbLLMClient()

    # llm_client._test_invoke()
    asyncio.run(llm_client._atest_invoke())

    print(llm_client.usage_statistics)
