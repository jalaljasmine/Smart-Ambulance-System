import pandas as pd
from models.predict_severity import predict
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent


def recommend_hospital(severity):

    hospitals = pd.read_csv(BASE_DIR / "datasets" / "hospitals.csv")

    score = []

    for index, row in hospitals.iterrows():
        s = 0
        if severity == "Slight Injury":

            if row["Trauma_Center"] == "Yes":
                s += 10

            s += row["ICU_Beds"] * 0.2
            s += row["Emergency_Staff"] * 0.2

        elif severity == "Serious Injury":

            if row["Trauma_Center"] == "Yes":
                s += 40

            s += row["ICU_Beds"] * 0.3
            s += row["Emergency_Staff"] * 0.3

        elif severity == "Fatal Injury":

            if row["Trauma_Center"] == "Yes":
                s += 60

            s += row["ICU_Beds"] * 0.4
            s += row["Emergency_Staff"] * 0.4

        score.append(s)

    hospitals["Score"] = score

    hospitals = hospitals.sort_values(
        by="Score",
        ascending=False
    ).reset_index(drop=True)

    best = hospitals.iloc[0]

    return best


if __name__ == "__main__":

    severity = predict(None)

    print("Predicted Severity:", severity)

    best = recommend_hospital(severity)

    print("\nBest Hospital")
    print(best["Hospital"])

    print("\nHospital Details")
    print(best)