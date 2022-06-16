import argparse

parser = argparse.ArgumentParser("Generating High Freq parser")
parser.add_argument("--load-dataset", type=str, default='./random_dataset_optimized_high_freq.xlsx', help="The path where the generated dataset will be stored")
args = parser.parse_args()

prefix = None
if "random_dataset_" in args.load_dataset:
	prefix = "random_set"
elif "train_dataset_" in args.load_dataset:
	prefix = "train_set"
elif "val_dataset_" in args.load_dataset:
	prefix = "val_set"

print(prefix)
print(f"{prefix}_kippo.xlsx")