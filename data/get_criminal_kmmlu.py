from datasets import load_dataset
from pathlib import Path
import pandas as pd

def main():
    ds = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
    df = pd.DataFrame(ds)
    out = Path(__file__).resolve().parent / "criminal_kmmlu.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()