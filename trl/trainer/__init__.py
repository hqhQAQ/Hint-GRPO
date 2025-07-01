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

from typing import TYPE_CHECKING

from ..import_utils import OptionalDependencyNotAvailable, _LazyModule, is_diffusers_available


_import_structure = {
    "callbacks": [
        "LogCompletionsCallback",
        "MergeModelCallback",
        "RichProgressCallback",
        "SyncRefModelCallback",
        "WinRateCallback",
    ],
    "dpo_config": ["DPOConfig", "FDivergenceConstants", "FDivergenceType"],
    "dpo_trainer": ["DPOTrainer"],
    "grpo_config": ["GRPOConfig"],
    "grpo_trainer": ["GRPOTrainer"],
    "grpo_vl_config": ["GRPOVLConfig"],
    "grpo_vl_trainer": ["GRPOVLTrainer"],
    "iterative_sft_trainer": ["IterativeSFTTrainer"],
    "judges": [
        "AllTrueJudge",
        "BaseBinaryJudge",
        "BaseJudge",
        "BasePairwiseJudge",
        "BaseRankJudge",
        "HfPairwiseJudge",
        "OpenAIPairwiseJudge",
        "PairRMJudge",
    ],
    "model_config": ["ModelConfig"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "utils": [
        "ConstantLengthDataset",
        "DataCollatorForCompletionOnlyLM",
        "RunningMoments",
        "compute_accuracy",
        "disable_dropout_in_model",
        "empty_cache",
        "peft_module_casting_to_bf16",
    ],
}
try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["ddpo_trainer"] = ["DDPOTrainer"]

if TYPE_CHECKING:
    from .callbacks import (
        LogCompletionsCallback,
        MergeModelCallback,
        RichProgressCallback,
        SyncRefModelCallback,
        WinRateCallback,
    )
    from .dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
    from .dpo_trainer import DPOTrainer
    from .gkd_config import GKDConfig
    from .gkd_trainer import GKDTrainer
    from .grpo_config import GRPOConfig
    from .grpo_trainer import GRPOTrainer
    from .iterative_sft_trainer import IterativeSFTTrainer
    from .judges import (
        AllTrueJudge,
        BaseBinaryJudge,
        BaseJudge,
        BasePairwiseJudge,
        BaseRankJudge,
        HfPairwiseJudge,
        OpenAIPairwiseJudge,
        PairRMJudge,
    )
    from .model_config import ModelConfig
    from .sft_config import SFTConfig
    from .sft_trainer import SFTTrainer
    from .utils import (
        ConstantLengthDataset,
        DataCollatorForCompletionOnlyLM,
        RunningMoments,
        compute_accuracy,
        disable_dropout_in_model,
        empty_cache,
        peft_module_casting_to_bf16,
    )

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .ddpo_trainer import DDPOTrainer
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
