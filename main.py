import argparse
from segment.line import run as run_line
from segment.character import run as run_char
from classification.resnet50 import run as run_model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Pipeline")

    # Line Segmentation
    parser.add_argument("--line_input", type=str, default="input", help="Path to the image file")
    parser.add_argument("--line_output", type=str, default="line-crops", help="Path to the image file")

    # Character Segmentation
    parser.add_argument("--char_input", type=str, default="line-crops", help="Path to the image file")
    parser.add_argument("--char_output", type=str, default="char-crops", help="Path to the image file")

    # Model
    parser.add_argument("--model_input", type=str, default="char-crops", help="Path to the image file")
    parser.add_argument("--model_output", type=str, default="output", help="Path to the image file")
    # parser.add_argument("--model_train_input", type=str, default="resnet50", help="Path to the image file")

    args = parser.parse_args()
    
    run_line(args)
    run_char(args)
    run_model(args)
