import argparse
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-2-9b-it")
    ap.add_argument("--word", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    for label, text, specials in [
        ("no_space", args.word, False),
        ("with_space", " " + args.word, False),
        ("no_space+specials", args.word, True),
        ("with_space+specials", " " + args.word, True),
    ]:
        ids = tok.encode(text, add_special_tokens=specials)
        print(f"{label:>18}: {ids} -> {[tok.decode([i]) for i in ids]} (len={len(ids)})")

if __name__ == "__main__":
    main()
