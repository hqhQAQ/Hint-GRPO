# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
import torch
import torch.utils.data
import transformers
import warnings
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer
from trl.trainer import ModelConfig

import copy
import deepspeed

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams


if is_wandb_available():
    import wandb
import torch.nn as nn
from torch.utils.data import Sampler
# New added
import random
from qwen_vl_utils import process_vision_info
from .utils import pad

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

SYSTEM_PROMPT = (
    "A conversation between User and Assistant."
    "The Assistant outputs the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
)

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class Qwen2VLGRPOVLLMTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        model_args: ModelConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # qwen2-vl related params
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        logger = None,
        num_groups: int = 3
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            # torch_dtype = model_init_kwargs.get("torch_dtype")
            torch_dtype = model_args.torch_dtype
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.use_vllm = args.use_vllm

        # rewrite the processing AutoTokenizer -> AutoProcessor
        model_id = model if isinstance(model, str) else model.config._name_or_path
        if processing_class is None:
            if "Qwen" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == "cuda"
                    and int(vllm_device.split(":")[1]) >= torch.cuda.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"cuda:{idx}" for idx in range(self.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )
                with world_size_patch, profiling_patch:
                    print("vllm is running on: ", vllm_device)
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=torch.bfloat16,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        enforce_eager=True,
                        max_model_len=args.max_completion_length,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = (
                0  # tag to avoid useless loading during grad accumulation
            )

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError(
                "Qwen2VLGRPOVLLMTrainer only supports vllm generation, please set --use_vllm True"
            )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )
        # New added
        self.logger = logger
        self.special_hint_ratios = [-3.0, -4.0]
        self.use_random = False
        self.num_groups = num_groups
        self.group_hint_ratios = [i/self.num_groups for i in range(self.num_groups)]

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # We need a custom sampler that samples the same prompt multiple times
    """
    def _get_train_sampler(self):
        return RepeatRandomSampler(self.train_dataset, self.num_generations)
    """

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        **inputs,
    ):
        logits = model(**inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = inputs['input_ids'][:,
                    1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _get_part_steps(self, reasoning_steps, ratio):
        part_len = int(len(reasoning_steps) * ratio)
        res = ' '.join(reasoning_steps[:part_len])
        return res

    def _map_to_msg(self, cur_input):
        prompt, reasoning_steps, image_path = cur_input["prompt"], cur_input["reasoning_steps"], cur_input["image_path"]
        hint_text = ""  # Do not add hint here
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path[0]},
                {"type": "text", "text": prompt["content"] + hint_text}
            ]}
        ]

    def _process_prompts_text_list(self, prompts_text, inputs):
        res = []
        num_groups = self.num_groups
        for group_idx in range(num_groups):
            res += [prompts_text[0] + "<think>" + self._get_part_steps(inputs[0]["reasoning_steps"], self.group_hint_ratios[group_idx])]
        return res

    def _process_prompts_text(self, prompt, cur_input, cur_idx):
        num_generations_per_group = self.num_generations
        group_idx = cur_idx // num_generations_per_group
        prompt += "<think>" + self._get_part_steps(cur_input["reasoning_steps"], self.group_hint_ratios[group_idx])
        return prompt

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        num_generations_per_group = self.num_generations
        if isinstance(inputs[0]["prompt"], dict):   # LLaVA-CoT dataset
            prompts = [self._map_to_msg(x) for x in inputs]
            for input_idx in range(len(inputs)):
                inputs[input_idx]["prompt"] = self._map_to_msg(inputs[input_idx])
        images = [x["image"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompts_texts = self._process_prompts_text_list(prompts_text, inputs)
        group_prompt_inputs = []
        for group_idx in range(self.num_groups):
            cur_prompt_inputs = self.processing_class(
                # prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                text=prompts_texts[group_idx : group_idx + 1],
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            group_prompt_inputs.append(
                super()._prepare_inputs(cur_prompt_inputs)
            )
        # manual batch
        batch_size = self.num_generations * self.num_groups

        group_prompt_lengths = [x["input_ids"].shape[1] for x in group_prompt_inputs]
        group_batched_inputs = []
        for group_idx in range(self.num_groups):
            group_batched_inputs.append({
                k: v.repeat(num_generations_per_group, *[1] * (v.dim() - 1)) if isinstance(v, torch.Tensor) else v
                for k, v in group_prompt_inputs[group_idx].items()
            })

        if self.max_prompt_length is not None:
            for group_idx in range(self.num_groups):
                group_batched_inputs[group_idx]["input_ids"] = group_batched_inputs[group_idx]["input_ids"][:, -self.max_prompt_length:]
                group_batched_inputs[group_idx]["attention_mask"] = group_batched_inputs[group_idx]["attention_mask"][:, -self.max_prompt_length:]

        inputs_vllm = []

        for messages in prompts:
            prompt = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            for i in range(batch_size):
                processed_prompt = self._process_prompts_text(prompt, inputs[0], cur_idx=i)
                image_data, _ = process_vision_info(messages)
                inputs_vllm.append({
                    "prompt": processed_prompt,
                    "multi_modal_data": {
                        "image": image_data
                    },
                })

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                # remove_hooks(model)
                unwrapped_model = self.accelerator.unwrap_model(model)
                if is_compiled_module(unwrapped_model):
                    state_dict = unwrapped_model._orig_mod.state_dict()
                else:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
            self._last_loaded_step = self.state.global_step
            # add_hooks(model)

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_inputs_vllm = gather_object(inputs_vllm)
        if self.accelerator.is_main_process:
            outputs = self.llm.generate(all_inputs_vllm, sampling_params=self.sampling_params, use_tqdm=False)
            completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
        else:
            completion_ids = [None] * len(all_inputs_vllm)

        # Broadcast the completions from the main process to all processes, ensuring each process receives its 
        # corresponding slice.
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * batch_size,
            (self.accelerator.process_index + 1) * len(prompts) * batch_size,
        )
        completion_ids = completion_ids[process_slice]

        # Pad the completions, and concatenate them with the prompts
        """
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_completion_ids = torch.cat([batched_inputs["input_ids"], completion_ids], dim=1)
        """
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        group_prompt_completion_ids = []
        for group_idx in range(self.num_groups):
            group_completion_ids = completion_ids[group_idx * num_generations_per_group : (group_idx + 1) * num_generations_per_group]
            cur_batched_inputs = group_batched_inputs[group_idx]
            cur_prompt_completion_ids = [torch.cat([cur_batched_inputs["input_ids"][idx], ids], dim=0) for idx, ids in enumerate(group_completion_ids)]
            group_prompt_completion_ids.extend(cur_prompt_completion_ids)
        group_prompt_completion_ids = pad(group_prompt_completion_ids, padding_value=self.processing_class.pad_token_id)

        # Mask everything after the first EOS token
        split_prompt_completion_ids = torch.chunk(group_prompt_completion_ids, self.num_groups, dim=0)
        group_completion_ids, group_completion_masks = [], []
        for group_idx, prompt_completion_ids in enumerate(split_prompt_completion_ids):
            completion_ids = prompt_completion_ids[:, group_prompt_lengths[group_idx]:]
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            group_completion_ids.append(completion_ids)
            group_completion_masks.append(completion_mask)

        # attention_mask = torch.cat([batched_inputs["attention_mask"], completion_mask], dim=1)
        attention_mask = []
        for group_idx in range(self.num_groups):
            cur_attenion_mask = torch.cat([group_batched_inputs[group_idx]["attention_mask"], group_completion_masks[group_idx]], dim=1)
            attention_mask.append(cur_attenion_mask)
        attention_mask = torch.cat(attention_mask, dim=0)

        def get_per_token_logps(model, **inputs):
            logits = model(**inputs).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = inputs['input_ids'][:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        # batched_inputs1 = batched_inputs.copy()
        batched_inputs1 = {}
        for key_name in ['pixel_values', 'image_grid_thw']:
            batched_inputs1[key_name] = torch.cat([x[key_name] for x in group_batched_inputs], dim=0)
        batched_inputs1["input_ids"] = group_prompt_completion_ids
        batched_inputs1["attention_mask"] = attention_mask
        group_per_token_logps = get_per_token_logps(model, **batched_inputs1)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)

        # per_token_logps = per_token_logps[:, prompt_length - 1:]
        split_per_token_logps = torch.chunk(group_per_token_logps, self.num_groups, dim=0)
        group_per_token_logps = []
        for group_idx, per_token_logps in enumerate(split_per_token_logps):
            per_token_logps = per_token_logps[:, group_prompt_lengths[group_idx] - 1:]
            group_per_token_logps.append(per_token_logps)

        with torch.inference_mode():
            if self.ref_model is not None:
                group_ref_per_token_logps = get_per_token_logps(self.ref_model, **batched_inputs1)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    group_ref_per_token_logps = get_per_token_logps(model, **batched_inputs1)
        # ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        split_ref_per_token_logps = torch.chunk(group_ref_per_token_logps, self.num_groups, dim=0)
        group_ref_per_token_logps = []
        for group_idx, ref_per_token_logps in enumerate(split_ref_per_token_logps):
            ref_per_token_logps = ref_per_token_logps[:, group_prompt_lengths[group_idx] - 1:]
            group_ref_per_token_logps.append(ref_per_token_logps)

        # Compute the KL divergence between the model and the reference model
        all_per_token_kl = []
        for group_idx in range(self.num_groups):
            ref_per_token_logps = group_ref_per_token_logps[group_idx]
            per_token_logps = group_per_token_logps[group_idx]
            all_per_token_kl.append(torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        # Decode the generated completions
        completions = []
        for completion_ids in group_completion_ids:
            completions.extend(self.processing_class.batch_decode(completion_ids, skip_special_tokens=True))
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations * self.num_groups)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_groups * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        split_rewards_per_func = torch.chunk(rewards_per_func, self.num_groups, dim=0)
        group_losses, all_rewards = [], []
        for group_idx in range(self.num_groups):
            # Sum the rewards from all reward functions
            rewards = split_rewards_per_func[group_idx].sum(dim=1)

            # Gather reward
            all_rewards.append(rewards)

            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

            # x - x.detach() allows for preserving gradients from x
            ref_per_token_logps = group_ref_per_token_logps[group_idx]
            per_token_logps = group_per_token_logps[group_idx]
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(per_token_loss - self.beta * all_per_token_kl[group_idx])
            completion_mask = group_completion_masks[group_idx]
            group_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            group_losses.append(group_loss)

        def find_first_positive(lst):
            try:
                return next(i for i, num in enumerate(lst) if num > 0)
            except StopIteration:
                return 0

        rewards_sum = [x.sum().item() for x in all_rewards]
        selected_group_idx = find_first_positive(rewards_sum)
        selected_rewards = all_rewards[selected_group_idx]
        loss = group_losses[selected_group_idx]

        # Log the metrics
        for group_idx in range(self.num_groups):
            self._metrics["reward_g{}".format(group_idx)].append(self.accelerator.gather_for_metrics(all_rewards[group_idx]).mean().item())
        self._metrics["reward_selected"].append(self.accelerator.gather_for_metrics(selected_rewards).mean().item())

        return loss

        
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()