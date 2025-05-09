import Levenshtein
import os

def run(args):
    assert os.path.exists(args.answers), "Answers path does not exist"

    for i in os.listdir("results"):
        model_output_path = os.path.join("results/", i)

        ground_truth_path = os.path.join(args.answers, "_".join(i.split("_")[0:2]) + ".txt")
        
        print("Evaluation for:", i.split(".")[0], "Model:", args.model)
        
        # Ground truth
        with open(ground_truth_path, "r", encoding='utf-8') as f:
            ground_truth = f.read()

        # Model answer
        with open(model_output_path, "r", encoding='utf-8') as f:
            model_answer = f.read()

        print(f"Lecvenshtein distance: {Levenshtein.distance(ground_truth, model_answer):.4f}")
        print(f"Levenshtein distance normalized: {Levenshtein.distance(ground_truth, model_answer) / max(len(ground_truth), len(model_answer)):.4f}")
        print(f"Levenshtein ratio: {Levenshtein.ratio(ground_truth, model_answer):.4f}\n")