# Public card-fraud data (experiment F4)

The raw file `creditcard.csv` is not shipped with this package. It is the public "Credit Card Fraud Detection" data set of the Machine Learning Group of ULB (Universite Libre de Bruxelles), described in Dal Pozzolo, Caelen, Johnson and Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE SSCI 2015 (doi 10.1109/SSCI.2015.33). It is available on OpenML as data set 1597 and on Kaggle as `mlg-ulb/creditcardfraud`.

Expected file: 284,807 rows and 31 columns (`Time`, `V1` to `V28`, `Amount`, `Class`), about 150 MB. `Class` = 1 for 492 rows. Any file with these row, column and fraud counts is the same data set. No SHA-256 of the original file is recorded in this package; if you need to compare copies, compute it with `sha256sum creditcard.csv`.

Place the file at `data/card_fraud/creditcard.csv` (relative to the package root) and run, from the package root:

    export PYTHONPATH=$PWD/src
    python3 scripts/prep_F4.py
    python3 scripts/run_F4.py smoke
    python3 scripts/run_F4.py run
    python3 scripts/run_F4.py rowboot
    python3 scripts/analyze_F4.py

(`PYTHONPATH=src python3 scripts/prep_F4.py` etc. is equivalent.) `prep_F4.py` needs pandas and scikit-learn.

Files in this folder:

| File | Content |
|---|---|
| `F4_calibration.json` | Calibration report written by `prep_F4.py`: rows and frauds per split, fixed-bin table, band share, zeta-hat and its bootstrap interval, ECE, AUC and AP (shipped) |
| `F4_scores.npz` | Calibrated scores, labels and amounts of the evaluation split (written by `prep_F4.py`; not shipped) |
| `creditcard.csv` | Raw data (not shipped) |
