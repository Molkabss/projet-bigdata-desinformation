"""
prepare_dataset.py
Fusionne Fake.csv + True.csv (Kaggle) en un seul fichier news.tsv,
au format attendu par clean_features.pig :
id, title, text, subject, date, label (Fake=0, Real=1), séparés par tabulation.

Usage :
    python prepare_dataset.py
Lit  : data/Fake.csv, data/True.csv
Ecrit: data/news.tsv
"""

import csv

def clean_field(s):
    """Supprime tabulations/retours à la ligne qui casseraient le format TSV."""
    if s is None:
        return ""
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

def load_csv(path, label, start_id):
    rows = []
    idx = start_id
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                idx,
                clean_field(row.get("title", "")),
                clean_field(row.get("text", "")),
                clean_field(row.get("subject", "")),
                clean_field(row.get("date", "")),
                label
            ))
            idx += 1
    return rows, idx

def main():
    fake_rows, next_id = load_csv("data/Fake.csv", label=0, start_id=1)
    true_rows, _ = load_csv("data/True.csv", label=1, start_id=next_id)

    all_rows = fake_rows + true_rows

    with open("data/news.tsv", "w", encoding="utf-8", newline="") as out:
        for (id_, title, text, subject, date, label) in all_rows:
            out.write(f"{id_}\t{title}\t{text}\t{subject}\t{date}\t{label}\n")

    print(f"{len(all_rows)} lignes écrites dans data/news.tsv")
    print(f"  Fake (0): {len(fake_rows)}")
    print(f"  Real (1): {len(true_rows)}")

if __name__ == "__main__":
    main()