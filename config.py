from rich import print

parameter = {
    "ROOT": r"/mnt/d/dataset",
    "dataset_csv": r"train.csv",
    "batch_size": 32,
    "num_workers" : 8,
    "learning_rate" : 5e-4,
    "val_size" : 0.2,
    "seed" : 12,
    "epochs":10,
    "num_classes" : 12,
    "in_channel" : 3,
    "image_size" : (256,256),
    "save_model" : "Effect"
}


if __name__ == "__main__":
    print("paramters:",parameter)
