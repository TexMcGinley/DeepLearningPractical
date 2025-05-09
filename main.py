import argparse
from segment.line import run as run_line
from segment.character import run as run_char
from classification.resnet50 import run as run_resnet50, train as train_resnet50
from classification.alexnet import run as run_alexnet, train as train_alexnet
from evaluate.evaluate import run as run_evaluate
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Pipeline")

    parser.add_argument("input", type=str, help="Path to the image folder")

    parser.add_argument("--model", type=str, default="resnet50", help="Specifies the model to use for classification. Options: 'resnet50' or 'alexnet'")

    parser.add_argument("--answers", type=str, help="Path to the folder with answers to run evalution script")

    args = parser.parse_args()
    
    run_line(args)
    run_char(args)

    if args.model == "alexnet":
        if not os.path.exists("models/alexnet_model.pth"):
            print("Alexnet model not found. Train the model or provide weights")
            exit(1)
        run_alexnet(args)
    elif args.model == "resnet50":
        if not os.path.exists("models/resnet50_model.pth"):
            print("ResNet50 model not found. Train the model or provide weights")
            exit(1)
        run_resnet50(args)

    if args.answers:
        run_evaluate(args)
