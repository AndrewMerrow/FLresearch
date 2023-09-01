import torch
import argparse


parser = argparse.ArgumentParser(description="Flower")

parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )

args = parser.parse_args()

for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

print(torch.backends.cudnn.is_available())

#device = torch.device(
#        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
#    )
#device = torch.device("cuda:0")
#print("Device: " + str(device))