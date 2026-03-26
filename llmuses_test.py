import torch
from llmuses.constants import DEFAULT_ROOT_CACHE_DIR
from llmuses.run import run_task

DATASET_ROOT = "/gemini/code/neb_assistant/data/llmuses"

your_task_cfg = {
    "model_args": {"revision": None, "precision": torch.float16, "device_map": "auto"},
    "template_type": "qwen",
    "generation_config": {"do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": 512},
    "dataset_args": {
        "arc": {"local_path": DATASET_ROOT + "/arc"},
        "bbh": {"local_path": DATASET_ROOT + "/bbh"},
        "ceval": {"local_path": DATASET_ROOT + "/ceval"},
        "cmmlu": {"local_path": DATASET_ROOT + "/cmmlu"},
        "competition_math": {"local_path": DATASET_ROOT + "/competition_math"},
        "general_qa": {"local_path": DATASET_ROOT + "/general_qa"},
        "gsm8k": {"local_path": DATASET_ROOT + "/gsm8k"},
        "hellaswag": {"local_path": DATASET_ROOT + "/hellaswag"},
        "mmlu": {"local_path": DATASET_ROOT + "/mmlu"},
        "race": {"local_path": DATASET_ROOT + "/race"},
        "trivia_qa": {"local_path": DATASET_ROOT + "/trivia_qa"},
        "truthful_qa": {"local_path": DATASET_ROOT + "/truthful_qa"}
    },
    "dry_run": False,
    "model": "/gemini/code/neb_assistant/models/qwen/Qwen2-VL-7B-Instruct",
    # "model": "/gemini/code/neb_assistant/output_qwen",
    "datasets": ["arc", "bbh", "ceval", "cmmlu", "competition_math", "general_qa", "gsm8k", "hellaswag", "mmlu", "race",
                 "trivia_qa", "truthful_qa"],
    "work_dir": DEFAULT_ROOT_CACHE_DIR,
    "outputs": DEFAULT_ROOT_CACHE_DIR,
    "mem_cache": False,
    "dataset_hub": "Local",
    "dataset_dir": "data",
    "stage": "all",
    "limit": 10,
    "debug": False
}

run_task(task_cfg=your_task_cfg)

