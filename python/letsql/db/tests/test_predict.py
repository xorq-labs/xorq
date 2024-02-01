import os

import pandas as pd
import xgboost as xgb
import struct
from sklearn.model_selection import train_test_split
from pandas.api.types import is_float_dtype

import letsql.db as db


def train_xgb(data, objective="reg:squarederror"):
    # Assuming 'data' is a pandas DataFrame with the specified columns
    features = ["carat", "depth", "x", "y", "z"]
    target = "price"

    # Split the data into features and target variable
    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Instantiate an XGBoost regressor
    model = xgb.XGBRegressor(objective=objective, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Return the trained model
    return model


def convert_and_save(input_model, output_file):
    model = xgb.Booster()
    model.load_model(input_model)
    tmp_file = output_file + ".gbdt_rs.mid"
    # extract base score
    try:
        with open(input_model, "rb") as f:
            model_format = struct.unpack("cccc", f.read(4))
            model_format = b"".join(model_format)
            if model_format == b"bs64":
                print("This model type is not supported")
            elif model_format != "binf":
                f.seek(0)
            base_score = struct.unpack("f", f.read(4))[0]
    except Exception as e:
        print("error: ", e)
        return 1

    if os.path.exists(tmp_file):
        print(
            "Intermediate file %s exists. Please remove this file or change your output file path"
            % tmp_file
        )
        return 1

    # dump json
    model.dump_model(tmp_file, dump_format="json")

    # add base score to json file
    try:
        with open(output_file, "w") as f:
            f.write(repr(base_score) + "\n")
            with open(tmp_file) as f2:
                for line in f2.readlines():
                    f.write(line)
    except Exception as e:
        print("error: ", e)
        os.remove(tmp_file)
        return 1

    os.remove(tmp_file)
    return 0


def test_predict_model(tmp_model_dir, data_dir):
    data = pd.read_csv(data_dir / "csv" / "diamonds.csv")

    model = train_xgb(data, "reg:linear")
    model_path = os.path.join(tmp_model_dir, "model.xgb")
    model.save_model(model_path)

    gdbt_model_path = os.path.join(tmp_model_dir, "gdbt.model")
    convert_and_save(model_path, gdbt_model_path)

    features = ["carat", "depth", "x", "y", "z"]
    data_path = os.path.join(tmp_model_dir, "input.csv")
    data[features].to_csv(data_path, index=False)

    db.register_csv("diamonds", data_path)
    query = f"""
    select predict_xgb('{gdbt_model_path}', 'reg:linear', carat, depth, x, y, z) from diamonds;
    """
    predictions = db.sql(query).execute()

    assert len(predictions) == len(data)
    assert is_float_dtype(predictions.squeeze())
