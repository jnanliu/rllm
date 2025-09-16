from rllm.data.dataset import DatasetRegistry

from datasets import Dataset


def prepare_data():
    all_train_datasets = [
        DatasetRegistry.load_dataset("dapo_math", "train"),
        DatasetRegistry.load_dataset("openscience2-20k", "train"),
        DatasetRegistry.load_dataset("reasoning-gym-27k", "train")
    ]

    all_test_datasets = [
        DatasetRegistry.load_dataset("aime2024", "test"),
        DatasetRegistry.load_dataset("gpqa-diamond", "test")
    ]

    train_dataset = []
    for dst in all_train_datasets:
        for example in dst:
            train_dataset.append(example)
    train_dataset = Dataset.from_list(train_dataset)

    test_dataset = []
    for dst in all_test_datasets:
        for example in dst:
            test_dataset.append(example)
    test_dataset = Dataset.from_list(test_dataset)

    train_dataset = DatasetRegistry.register_dataset("agentic-learning-6k", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("agentic-learning-test", test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())
