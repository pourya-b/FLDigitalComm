# %% Main function
import os
import sys
import numpy as np
from utils.config import Parameter_config, set_seed
from utils.trainer import Trainer

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

os.chdir(parent_dir)
os.environ["PROJECT_ROOT"] = parent_dir

def main(args):
    seed_num = 10  # number of repeats with different seeds

    for _ in range(seed_num):
        seed = np.random.randint(1000)
        print(f"seed: {seed}")
        set_seed(seed)

        instance = Trainer(args)
        instance.run()  # runs training and testing

if __name__ == "__main__":
    configer = Parameter_config(parent_dir)
    main(configer.args)
