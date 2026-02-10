import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/raw/monthly")

FILES = [
    "12_2024.xlsx","11_2024.xlsx","10_2024.xlsx","9_2024.xlsx","8_2024.xlsx","7_2024.xlsx",
    "7-8_2025.xlsx","6_2024.xlsx","5_2024.xlsx","5-6_2025.xlsx","4_2024.xlsx","3_2024.xlsx",
    "3-4_2025.xlsx","2_2024.xlsx","1-2_2025.xlsx","1_2024.xlsx",
]

dfs = []
for fname in tqdm(FILES, desc="Reading Excel files"):
    path = DATA_DIR / fname
    if not path.exists():
        print(f"Skipping missing file: {fname}")
        continue
    df = pd.read_excel(path, sheet_name="DATA", header=5)
    df["__source_file"] = fname
    df = df.dropna(how="all")
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# ---- Make parquet-safe: for object cols, try numeric else string ----
converted_numeric, converted_string = [], []

for col in combined_df.columns:
    if combined_df[col].dtype == "object":
        s = combined_df[col]

        # Try to convert to numeric if it mostly looks numeric
        num = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()
        numeric_ratio = (num.notna().sum() / non_null) if non_null else 0

        if numeric_ratio >= 0.95 and non_null > 0:
            combined_df[col] = num
            converted_numeric.append(col)
        else:
            combined_df[col] = s.astype("string")
            converted_string.append(col)

print(f"Converted object->numeric: {len(converted_numeric)} cols")
print(f"Converted object->string:  {len(converted_string)} cols")

output_path = Path("data/raw") / "full2024_8month2025.parquet"
combined_df.to_parquet(output_path, index=False)

print("Saved to:", output_path)
print("Shape:", combined_df.shape)
