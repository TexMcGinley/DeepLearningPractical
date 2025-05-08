import Levenshtein
import os

def run(args):
    for i in os.listdir(args.model_output):
        model_output_path = os.path.join(args.model_output, i)
        ground_truth_path = os.path.join(args.answers, i)
        
        print("Evaluation for:", i.split(".")[0], "Model:", args.model)
        # Ground truth
        with open(ground_truth_path, "r", encoding='utf-8') as f:
            ground_truth = f.read()

        # Model answer
        with open(model_output_path, "r", encoding='utf-8') as f:
            model_answer = f.read()

        print("Lecvenshtein distance:", Levenshtein.distance(ground_truth, model_answer))
        print("Levenshtein distance normalized:", Levenshtein.distance(ground_truth, model_answer) / max(len(ground_truth), len(model_answer)))
        print("Levenshtein ratio:", Levenshtein.ratio(ground_truth, model_answer))
        print("\n")