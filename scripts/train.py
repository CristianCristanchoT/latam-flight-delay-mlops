"""Train the delay model and save it to challenge/model.xgb."""
import pandas as pd
from challenge.model import DelayModel

if __name__ == "__main__":
    data = pd.read_csv("data/data.csv")
    model = DelayModel()
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)
    print("Model trained and saved to challenge/model.xgb")
