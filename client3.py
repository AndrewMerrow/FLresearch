import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings
import copy
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split

    def set_parameters(self, parameters):
        """Loads a CNN model and replaces it parameters with the ones
        given."""
        #print("Params: " + str(parameters))
        #model = utils.load_efficientnet(classes=10)
        model = utils.Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        print("\nBatch size: " + str(config['batch_size']))
        print("Current round: " + str(config['current_round']))
        print("Local epochs: " + str(config['local_epochs']))
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )

        idxs = (self.testset.targets == 5).nonzero().flatten().tolist()
        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)
        
        #create a copy to be poisoned and another copy as a control 
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(self.trainset), idxs)
        clean_val_set = utils.DatasetSplit(copy.deepcopy(self.testset), idxs)

        #utils.poison_dataset(poisoned_val_set.dataset, idxs, poison_all=True)
        #print(poisoned_val_set.dataset.data.shape)

        poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=256, shuffle=False, pin_memory=False)
        
        #test images for visualization to confirm poisoning was successful 
        test_poison = poisoned_val_set.dataset.data[49988]
        #test_clean = clean_val_set.dataset.data[3000]
        test_clean = poisoned_val_set.dataset.data[49988]

        #visualize the poisoned image
        #fig = plt.figure()
        #ax1 = fig.add_subplot(2,2,1)
        #ax1.imshow(test_clean)
        #ax2 = fig.add_subplot(2,2,2)
        #ax2.imshow(test_poison)
        #plt.show()

        #training
        results = utils.train(model, trainLoader, valLoader, poisoned_val_loader, epochs, self.device)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=256)

        loss, accuracy = utils.test(model, testloader, steps, self.device)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run(device: str = "cpu"):
    """Weak tests to check whether all client methods are working as
    expected."""

    #model = utils.load_efficientnet(classes=10)
    model = utils.Net()
    trainset, testset = utils.load_partition(0)
    idxs = (trainset.targets == 5).nonzero().flatten().tolist()
    #print(idxs)
    utils.poison_dataset(trainset, idxs, poison_all=True)
    #trainset = torch.utils.data.Subset(trainset, range(10))
    #testset = torch.utils.data.Subset(testset, range(10))
    client = CifarClient(trainset, testset, device)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 16, "local_epochs": 1, "current_round": 1},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})

    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--dry",
        type=bool,
        default=False,
        required=False,
        help="Do a dry-run to check the client",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--poison",
        type=bool,
        default=False,
        required=False,
        help="Set to true to make the client poison their train data"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda:2" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    if args.dry:
        client_dry_run(device)
    else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        trainset, testset = utils.load_partition(args.partition)

        if args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))

        if args.poison:
            idxs = (trainset.targets == 5).nonzero().flatten().tolist()
            utils.poison_dataset(trainset, idxs, poison_all=True)

        # Start Flower client
        client = CifarClient(trainset, testset, device)

        fl.client.start_numpy_client(server_address="10.100.116.10:8080", client=client)


if __name__ == "__main__":
    main()