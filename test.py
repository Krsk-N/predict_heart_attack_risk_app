import pandas as pd
from sklearn.metrics import classification_report
import argparse


def main(args):
    corr_df = pd.read_csv(args.correct, index_col=0)
    stud_df = pd.read_csv(args.student, index_col=0)
    assert list(stud_df.columns) == ["id", "prediction"]
    assert len(stud_df) == len(corr_df)
    print(classification_report(corr_df["prediction"], stud_df["prediction"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test script")
    parser.add_argument("--student", type=str, required=True, help="path to students answer")
    parser.add_argument("--correct", type=str, default="correct_answers.csv", help="path to correct answers")
    arguments = parser.parse_args()
    main(arguments)
