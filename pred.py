from Model_Prep import prep_input, run_model
from RandomForest import load_forest_from_python_dict
import pandas as pd
import model_parms

td = model_parms.td
rf_dict = model_parms.rf
rf = load_forest_from_python_dict(rf_dict)


def open_data(filename):

    df = pd.read_csv(filename)

    return df


def predict_all(filename):

    X = open_data(filename)

    X = prep_input(X.copy(), td)

    output = run_model(rf, X, td)

    return output
