import json
from datasets import load_dataset, Dataset


if __name__ == "__main__":
    examples = []
    with open("/mnt/shared-storage-user/liujunnan/datasets/qwen3_short_cot-parsed.jsonl") as f:
        for line in f:
            example = json.loads(line)
            example["trajectory"] = example["trajectory"].replace("<think>\nNone\n</think>\n", "").replace("<think>\nNone\n</think>", "")
            examples.append(example)

    with open("/mnt/shared-storage-user/liujunnan/datasets/qwen3_short_cot-parsed.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    datasets = []
    for example in examples:
        datasets.append(
            {
                "prompt": example["question"],
                "response": example["trajectory"]
            }
        )

    dataset = Dataset.from_list(datasets)
    dataset.to_parquet("/mnt/shared-storage-user/liujunnan/datasets/qwen3_instruct_disitll.parquet")