# called by main.py
import argparse
import torch
import numpy as np
import random
import os
import json


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Parameter_config:
    def __init__(self):
        self.parse_args()
        self.parse_hyperparameters()
        self.update_index = int(min(self.hyperparameters.keys()))

    def get_search_size(self):
        return len(list(self.hyperparameters.keys()))

    def update_args(self, seed, seed_idx):
        self.args.seed = seed
        self.args.seed_idx = seed_idx
        self.args = (self.default_args)
        self.args.run_index = self.update_index
        hyper_param_dict = self.hyperparameters[str(self.update_index)]
        hyper_param_length = len(list(hyper_param_dict.keys()))
        self.update_index += 1

        for i in range(hyper_param_length):
            self.args.__dict__[list(hyper_param_dict.keys())[i]] = list(hyper_param_dict.values())[i]

    def reset(self):
        self.update_index = int(min(self.hyperparameters.keys()))

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Federated Learning Under a Digital Communication Model")
        parser.add_argument("--name", type=str, default="digitFL", help="name of the run")
        parser.add_argument("-user_num", default=5, type=int, help="Number of agents")
        parser.add_argument("-noise_level", default=22, type=float, help="SNR in dB")
        parser.add_argument("-QAM_size", default=16, type=float, help="QAM constellation size")
        parser.add_argument("-dynamic_QAM", default=False, type=bool, help="dynamic QAM constellation flag")
        parser.add_argument("-vip_length", default=8, type=int, help="number of bits sent with low BER")
        parser.add_argument("-precision", default=32, type=float, help="bit precision")  # HARD CODED in this version
        parser.add_argument("-communication", default=True, type=bool, help="if digital communication takes place. Default is digital otherwise it is analog")
        parser.add_argument("-VIP", default=True, type=bool, help="a flag for VIP bits in each packet sent with low BER")
        parser.add_argument("--epochs", default=10, type=int, help="Number of Epochs for main training")
        parser.add_argument("--batch_size", default=50, type=int, help="Batch size for training")
        parser.add_argument("--learning_rate", default=1, type=float, help="Learning Rate for main training")
        parser.add_argument("-aggregation_intv", default=5, type=int, help="Interval between two aggregations in FL")
        parser.add_argument("--run_index", default=0, type=int, help="Run Index")
        parser.add_argument("--save_model", default=True, type=bool, help="If the trained model should be saved")
        self.default_args = parser.parse_args()

        self.default_args.path = os.path.dirname(os.path.abspath(__file__)) + '/'
        with open(self.default_args.path + 'commandline_args.txt', 'w') as f:
            json.dump(self.default_args.__dict__, f, indent=2)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.default_args.n_gpu = 1
        self.default_args.device = "cpu"  # torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        print("Using Device: ", self.default_args.device)
        print("GPU Count: ", torch.cuda.device_count())
        print("Visible Device Names: ", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)

        self.args = self.default_args
        return self.default_args

    def parse_hyperparameters(self):  # set parameters here for different runs. The code uses the defaults above, unless otherwise specified below
        self.hyperparameters = {  # modify accordingly
            "1": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 4, "communication": True, "dynamic_QAM": False},
            "2": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 8, "communication": True, "dynamic_QAM": False},
            "3": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 16, "communication": True, "dynamic_QAM": False},
            "4": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 32, "communication": True, "dynamic_QAM": False},
            "5": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 32, "communication": True, "dynamic_QAM": True},
            "6": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 32, "communication": False, "dynamic_QAM": False},
        }
        return self.hyperparameters
