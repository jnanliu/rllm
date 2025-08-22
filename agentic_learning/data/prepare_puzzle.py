from functools import partial

import reasoning_gym
from datasets import Dataset
from tqdm import tqdm

from rllm.data.dataset import DatasetRegistry


configs = lambda is_train, size: [
    ("ab", 1.0, {"seed": 42, "length": 10, "size": size}),
    ("ab", 1.0, {"seed": 42, "length": 15, "size": size}),
    ("acre", 1.0, {"seed": 42, "train": 1 if is_train else 0, "size": size}),
    ("advanced_geometry", 1.0, {"seed": 42, "min_coord": -100, "max_coord": 100, "size": size}),
    ("aiw", 1.0, {"seed": 42, "max_entities": 10, "size": size}),
    ("cryptarithm", 1.0, {"seed": 42, "min_words": 5, "max_words": 20, "size": size}),
    ("dice", 1.0, {"seed": 42, "num_dice": 5, "max_dice_size": 30, "size": size}),
    ("futoshiki", 1.0, {"seed": 42, "size": size}),
    ("game_of_life", 1.0, {"seed": 42, "grid_size_x": 30, "grid_size_y": 30, "simulation_steps": 3, "size": size}),
    ("game_of_life", 1.0, {"seed": 42, "grid_size_x": 30, "grid_size_y": 30, "simulation_steps": 4, "size": size}),
    ("game_of_life", 1.0, {"seed": 42, "grid_size_x": 30, "grid_size_y": 30, "simulation_steps": 5, "size": size}),
    ("game_of_life_halting", 1.0, {"seed": 42, "grid_size_x": 30, "grid_size_y": 30, "difficulty": 3, "num_oscillators": 8, "max_simulation_steps": 40, "size": size}),
    ("jugs", 1.0, {"seed": 42, "difficulty": 20, "size": size}),
    ("knight_swap", 1.0, {"seed": 42, "size": size}),
    ("knights_knaves", 1.0, {"seed": 42, "n_people": 3, "depth_constraint": 3, "width_constraint": 3, "size": size}),
    ("knights_knaves", 1.0, {"seed": 42, "n_people": 5, "depth_constraint": 5, "width_constraint": 5, "size": size}),
    ("mahjong_puzzle", 1.0, {"seed": 42, "min_num_rounds": 30, "size": size}),
    ("needle_haystack", 1.0, {"seed": 42, "min_num_statements": 50, "size": size}),
    ("quantum_lock", 1.0, {"seed": 42, "difficulty": 10, "size": size}),
    ("quantum_lock", 1.0, {"seed": 42, "difficulty": 20, "size": size}),
    ("rush_hour", 1.0, {"seed": 42, "min_moves": 10, "size": size}),
    ("self_reference", 1.0, {"seed": 42, "difficulty": 10, "size": size}),
    ("sudoku", 1.0, {"seed": 42, "size": size}),
    ("zebra_puzzles", 1.0, {"seed": 42, "num_people": 4, "num_characteristics": 4, "size": size}),
    ("zebra_puzzles", 1.0, {"seed": 42, "num_people": 5, "num_characteristics": 5, "size": size}),
    ("zebra_puzzles", 1.0, {"seed": 42, "num_people": 6, "num_characteristics": 6, "size": size}),("zebra_puzzles", 1.0, {"seed": 42, "num_people": 7, "num_characteristics": 7, "size": size})
]

def prepare_puzzle_data():
    train_datasets = [
        reasoning_gym.create_dataset(name, **config)
        for name, _, config in configs(True, 1000)
    ]
    lst = []
    for d in tqdm(train_datasets):
        for item in tqdm(d, total=1000):
            item.pop("metadata")
            lst.append(item)
    train_dataset = Dataset.from_list(lst)

    test_datasets = [
        reasoning_gym.create_dataset(name, **config)
        for name, _, config in configs(False, 10)
    ]
    lst = []
    for d in tqdm(test_datasets):
        for item in tqdm(d, total=10):
            item.pop("metadata")
            lst.append(item)
    test_dataset = Dataset.from_list(lst)

    def preprocess_fn(example, idx, split: str = "train"):
        return {
            "question": example["question"],
            "ground_truth": example["answer"],
            "split": split,
            "data_source": "puzzle",
        }

    train_dataset = train_dataset.map(partial(preprocess_fn, split="train"), with_indices=True)
    test_dataset = test_dataset.map(partial(preprocess_fn, split="test"), with_indices=True)

    train_dataset = DatasetRegistry.register_dataset("reasoning-gym-27k", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("reasoning-gym-2_7k", test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    prepare_puzzle_data()