import argparse
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def parse_args():
    # parse cli arguments for flexible usage
    parser = argparse.ArgumentParser(description="run ner inference to extract mountain names")
    parser.add_argument(
        "--text",
        type=str,
        default="The climbers reached the summit of Everest at dawn.",
        help="input string to test model on"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "mountain_ner_model"),
        help="path to local folder with saved model and tokenizer"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # sanity check if model folder actually exists before loading
    if not os.path.exists(args.model_path):
        print(f"error: model folder not found at path '{args.model_path}'")
        return

    # load fine-tuned model and tokenizer from local checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    # setup hf ner pipeline with simple aggregation to merge b- and i- tokens
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    # run prediction on provided text
    results = ner_pipeline(args.text)

    # extract words identified as mountain entities
    mountains = [res["word"].strip() for res in results if res.get("entity_group") == "MOUNTAIN"]

    # print clean formatted output
    print("\n" + "=" * 60)
    print(f"input text: {args.text}")
    print("-" * 60)

    if mountains:
        print(f"detected mountains: {', '.join(mountains)}")
    else:
        print("detected mountains: none")

    print("=" * 60)

    # print details for debugging score/entities
    if results:
        print("\ndetailed pipeline output:")
        for res in results:
            print(f" - entity: {res['entity_group']}, word: '{res['word']}', confidence: {res['score']:.4f}")
    print()


if __name__ == "__main__":
    main()