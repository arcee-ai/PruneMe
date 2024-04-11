OUTPUT_PATH = "./merged"  # folder to store the result in
LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
CONFIG_YML = "./slice.yaml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

# actually do merge
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

run_merge(
    merge_config,
    out_path=OUTPUT_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
    ),
)
print("Done!")
