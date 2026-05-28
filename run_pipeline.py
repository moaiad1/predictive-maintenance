import sys
sys.path.insert(0, "src")

from preprocess import run_preprocessing
from train import run_training
from evaluate import run_evaluation

if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: Preprocessing")
    print("=" * 50)
    run_preprocessing()

    print("\n" + "=" * 50)
    print("STEP 2: Training")
    print("=" * 50)
    run_training()

    print("\n" + "=" * 50)
    print("STEP 3: Evaluation")
    print("=" * 50)
    run_evaluation()

    print("\nPipeline complete.")
