# called by main.py
import random
import os
import json
import numpy as np
import argparse
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Parameter_config:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Federated Learning Under a Digital Communication Model")
        parser.add_argument("--name", type=str, default="digitFL", help="name of the run")
        parser.add_argument("--user_num", default=10, type=int, help="Number of agents")
        parser.add_argument("--noise_level", default=15, type=float, help="SNR in dB")
        parser.add_argument("--QAM_size", default=16, type=float, help="QAM constellation size")
        parser.add_argument("--dynamic_QAM", default=False, type=bool, help="dynamic QAM constellation flag")
        parser.add_argument("--vip_length", default=8, type=int, help="number of bits sent with low BER")
        parser.add_argument("--precision", default=32, type=float, help="bit precision")  # HARD CODED in this version
        parser.add_argument("--communication", default=False, type=bool, help="if digital communication takes place. Default is digital otherwise it is analog")
        parser.add_argument("--VIP", default=True, type=bool, help="a flag for VIP bits in each packet sent with low BER")
        parser.add_argument("--epochs", default=5, type=int, help="Number of Epochs for main training")
        parser.add_argument("--batch_size", default=50, type=int, help="Batch size for training")
        parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning Rate for main training")
        parser.add_argument("--aggregation_intv", default=5, type=int, help="Interval between two aggregations in FL")
        parser.add_argument("--run_index", default=0, type=int, help="Run Index")
        parser.add_argument("--save_model", default=True, type=bool, help="If the trained model should be saved")
        parser.add_argument("--log_dir", default="logs", type=str, help="The directory for saving the results/logs relative to the parent_dir")
        self.args = parser.parse_args()

        self.args.log_dir = os.path.join(self.parent_dir, self.args.log_dir)
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        arg_save_path = os.path.join(self.args.log_dir, 'config_arguments.txt')
        with open(arg_save_path, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.args.n_gpu = 1
        self.args.device = "cpu"  # torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        print("Using Device: ", self.args.device)
        print("GPU Count: ", torch.cuda.device_count())
        print("Visible Device Names: ", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)

        return self.args

        ## set parameters here for different runs. The code uses the defaults above, unless otherwise specified below
        ## modify accordingly in the commandline. Example: python main.py --epochs=10
        ## hyperparameter search examples:
        # 
        #     "1": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 4, "communication": False, "dynamic_QAM": False},
        #     "2": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 8, "communication": True, "dynamic_QAM": False},
        #     "3": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 16, "communication": True, "dynamic_QAM": False},
        #     "4": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 32, "communication": True, "dynamic_QAM": False},
        #     "5": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 32, "communication": True, "dynamic_QAM": True},
        #     "6": {"user_num": 10, "epochs": 5, "noise_level": 15, "learning_rate": 0.0005, "QAM_size": 32, "communication": False, "dynamic_QAM": False},
