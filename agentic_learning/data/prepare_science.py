from pathlib import Path
import random
random.seed(42)

from datasets import load_dataset
from functools import partial

from rllm.data.dataset import DatasetRegistry


def prepare_science_data():
    train_dataset = load_dataset(
        "nvidia/OpenScienceReasoning-2", 
        split="train", 
        cache_dir=Path(__file__).parents[2].joinpath(".cache")
    )
    train_dataset = train_dataset.select(random.sample(list(range(len(train_dataset))), 20_000))
    test_dataset = load_dataset("fingertap/GPQA-Diamond", split="test")

    def preprocess_fn(example, idx, split: str = "train"):
        return {
            "question": example["input"].replace("Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}.", "").strip() if split == "train" else example["question"],
            "ground_truth": example["expected_answer"] if split == "train" else example["answer"],
            "split": split,
            "data_source": "science",
        }

    train_dataset = train_dataset.map(partial(preprocess_fn, split="train"), with_indices=True)
    train_dataset = train_dataset.remove_columns([
        col for col in train_dataset.column_names 
        if col not in ["question", "ground_truth", "split", "data_source"]
    ])
    test_dataset = test_dataset.map(partial(preprocess_fn, split="test"), with_indices=True)
    test_dataset = test_dataset.remove_columns([
        col for col in test_dataset.column_names 
        if col not in ["question", "ground_truth", "split", "data_source"]
    ])

    train_dataset = DatasetRegistry.register_dataset("opensicence2-20k", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("gpqa-diamond", test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_science_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())