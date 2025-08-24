import argparse
from src.pipeline import train_pipeline
from src.deployment import inference

def main():
    parser = argparse.ArgumentParser(description='Brain cancer detection ML project')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help='Mode: train or predict')
    parser.add_argument('--image', type=str, help='Path to input image for prediction')

    args = parser.parse_args()

    if args.mode == 'train':
        train_pipeline.train()
    elif args.mode == 'predict':
        if not args.image:
            print("Please provide --image argument for prediction mode")
        else:
            pred_class, pred_probs = inference.predict(args.image)
            print(f"Predicted class index: {pred_class}")
            print(f"Class probabilities: {pred_probs}")

if __name__ == "__main__":
    main()
