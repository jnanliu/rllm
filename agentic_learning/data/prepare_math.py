from datasets import load_dataset
from functools import partial

from rllm.data.dataset import DatasetRegistry


def prepare_math_data():
    train_dataset = load_dataset("zwhe99/DeepMath-103K", split="train")
    train_dataset = train_dataset.filter(lambda x: x["difficulty"] >= 7)
    test_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    def preprocess_fn(example, idx, split: str = "train"):
        return {
            "question": example["question"] if split == "train" else example["problem"],
            "ground_truth": example["final_answer"] if split == "train" else example["answer"],
            "split": split,
            "data_source": "math",
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

    train_dataset = DatasetRegistry.register_dataset("deepmath_lv7-9", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())