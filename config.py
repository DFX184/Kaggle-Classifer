from rich import print
parameter = {
    "ROOT": r"D:\dataset",
    "dataset_csv": r"train.csv",
    "batch_size": 32,
    "device" : "cuda:0",
    "learning_rate" : 1e-3,
    "val_size" : 0.9,
    "seed" : 12,
    "epochs":10,
    "num_classes" : 12,
    "in_channel" : 3
}


if __name__ == "__main__":
    print("paramters:",parameter)