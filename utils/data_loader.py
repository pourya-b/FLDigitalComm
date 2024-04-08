# called by deep_network.py
from torch.utils.data import DataLoader, Dataset


class myDataset(Dataset):
    def __init__(self, dataset, user_num):
        # for evenly distributed data between the agents
        # self.image = dataset[0]
        # self.labels = dataset[1]

        # adjust accordingly for different datasets
        if user_num == 5:
            self.image = dataset[0][0:11000]  # for 5 agents
            self.labels = dataset[1][0:11000]
        elif user_num == 10 or user_num == 20 or user_num == 40:
            self.image = dataset[0][0:4000]  # for 10 or 20 agents
            self.labels = dataset[1][0:4000]

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, item):
        return self.image[item], self.labels[item]


def get_data_loaders(batch_size, user_num, train_dataset):
    dataset_train_data = train_dataset.train_data
    dataset_train_label = train_dataset.train_labels

    train_loader = {}

    # for evenly distributed data between the agents
    # adjust accordingly for different datasets
    # for l, i in enumerate(range(0, user_num, 1)):
    #     train = myDataset((dataset_train_data[int((60000/user_num) * l):int((60000/user_num) * (l+1))], dataset_train_label[int((60000/user_num) * l):int((60000/user_num) * (l+1))]))
    #     train_loader[str(l)] = DataLoader(train, batch_size=batch_size, shuffle=True)

    if user_num == 5:
        for l, i in enumerate(range(0, 10, 2)):  # for 5 agents
            tensor_1 = dataset_train_label == i
            tensor_2 = dataset_train_label == i+1
            train = myDataset((dataset_train_data[tensor_1 | tensor_2], dataset_train_label[tensor_1 | tensor_2]), user_num)
            train_loader[str(l)] = DataLoader(train, batch_size=batch_size, shuffle=True)

    elif user_num == 10 or user_num == 20 or user_num == 40:
        for i in range(user_num):  # for user_num agents
            train = myDataset((dataset_train_data[dataset_train_label == int(i/(user_num/10))], dataset_train_label[dataset_train_label == int(i/(user_num/10))]), user_num)
            train_loader[str(i)] = DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader
