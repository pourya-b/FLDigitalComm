# called by main.py
from deep_network import DeepNet
from torchvision import datasets
from torchvision.transforms import ToTensor


class Trainer:
    def __init__(self, args):
        self.args = args

        print("\n", 15 * "-", "Generating data", 15 * "-")
        print(f"# Run Index: {args.run_index}")
        train_data, test_data = self.generate_dataset(self.args)
        self.net = DeepNet(args, train_data, test_data)

    def generate_dataset(self, args):  # modify datasets accordingly
        train_data = datasets.MNIST(
            root='../MNIST_data',
            train=True,
            transform=ToTensor(),
            download=True,
        )
        test_data = datasets.MNIST(
            root='../MNIST_data',
            train=False,
            transform=ToTensor()
        )

        return train_data, test_data

    def run(self):  # runs the main training
        self.net.train()
        self.net.test()
