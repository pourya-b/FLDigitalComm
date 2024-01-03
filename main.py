# %% Main function
from config import Parameter_config, set_seed
from Net_Trainer import Trainer
import os
import numpy as np


def main():
    path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.chdir(path)

    seed_num = 10  # number of repeats with different seeds
    configer = Parameter_config()
    runs_num = configer.get_search_size()  # set different parameters and their default values in config.py

    for s in range(seed_num):
        seed = np.random.randint(1000)
        print(f"seed: {seed}")
        for i in range(runs_num):
            set_seed(seed)
            configer.update_args(seed, s)

            instance = Trainer(configer.args)
            instance.run()  # runs training and testing
        configer.reset()


if __name__ == "__main__":
    main()
