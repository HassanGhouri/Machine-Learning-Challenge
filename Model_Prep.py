import numpy as np
from RandomForest import RandomForest, load_forest_from_python_dict
import pandas as pd
import model_parms


def open_data(filename):

    df = pd.read_csv("cleaned_data_combined.csv")

    X = df.drop(columns=["Label"])
    y = df["Label"].values

    return X, y

def data_split(X, y):
    n_samples = X.shape[0]
   #np.random.seed(0)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_index = int(n_samples * 0.8)

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def prep_one_hot_encode_inplace(df, column):

    all_cats = set()
    for val in df[column].dropna():

        for piece in val.split(','):
            piece = piece.strip()

            if piece:
                all_cats.add(piece)

    for cat in all_cats:
        new_col = f"{column}__{cat}"
        df[new_col] = df[column].fillna("").apply(
            lambda x: int(cat in [c.strip() for c in x.split(',')])
        )

    df.drop(columns=[column], inplace=True)

def transform_one_hot_encode_inplace(df, column, all_ohe_cols):

    for col_name in all_ohe_cols:

        prefix, sep, cat_val = col_name.partition("__")

        if prefix == column:
            df[col_name] = df[column].fillna("").apply(
                lambda x: int(cat_val in [w.strip() for w in x.split(',')])
            )

    if column in df.columns:
        df.drop(columns=[column], inplace=True)


def build_model(X, y, num_estimators: int, max_depth: int, min_sample_split, bow_size: int):

    numeric_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)"]  # numeric
    cat_cols = ["Q3: In what setting would you expect this food to be served? Please check all that apply",
                "Q7: When you think about this food item, who does it remind you of?",
                "Q8: How much hot sauce would you add to this food item?"]  # multiple choice
    text_cols = ["Q2: How many ingredients would you expect this food item to contain?",
                 "Q4: How much would you expect to pay for one serving of this food item?",
                 "Q5: What movie do you think of when thinking of this food item?",
                 "Q6: What drink would you pair with this food item?"]  # sentence/short text
    label_col = "Label"

    X = X.drop(columns=["id"])
    X["combined_text"] = X[text_cols].apply(lambda row: " ".join(row.astype(str)), axis=1)

    X = X.drop(columns=text_cols)

    for cat_col in cat_cols:
        prep_one_hot_encode_inplace(X, cat_col)

    text_series = X["combined_text"].fillna("").astype(str)
    N = len(text_series)

    word_doc_freq = {}

    for text in text_series:
        words_in_doc = set(text.lower().split())

        for w in words_in_doc:

            if w in word_doc_freq:
                word_doc_freq[w] += 1

            else:
                word_doc_freq[w] = 1

    idf_dict = {}

    for w, df_count in word_doc_freq.items():
        idf_dict[w] = np.log(N / (1.0 + df_count))

    words_sorted_by_freq = sorted(word_doc_freq.items(), key=lambda x: x[1], reverse=True)
    top_words = [w for w, _ in words_sorted_by_freq[:bow_size]]

    word_index = {w: i for i, w in enumerate(top_words)}

    X_text = np.zeros((N, bow_size), dtype=float)

    for i, text in enumerate(text_series):

        counts = {}

        for w in text.lower().split():

            if w in word_index:
                counts[w] = counts.get(w, 0) + 1

        for w, tf in counts.items():
            j = word_index[w]
            X_text[i, j] = tf * idf_dict[w]

    row_norms = np.sqrt(np.sum(X_text ** 2, axis=1, keepdims=True))
    nonzero_mask = (row_norms[:, 0] != 0)
    X_text[nonzero_mask] /= row_norms[nonzero_mask]

    X.drop(columns=["combined_text"], inplace=True)

    ohe_cols = [c for c in X.columns if "__" in c]
    X_ohe = X[ohe_cols].values

    X_numeric = X[numeric_cols].values

    X_preprocessed = np.hstack([X_ohe, X_text, X_numeric])

    X_preprocessed_dense = X_preprocessed.astype(float)

    unique_labels = np.unique(y)
    label_dict = {lab: i for i, lab in enumerate(unique_labels)}
    y_encoded = np.array([label_dict[lbl] for lbl in y])

    rf = RandomForest(n_trees=num_estimators, max_depth=max_depth, min_samples_split=min_sample_split, n_feature=None)
    rf.fit(X_preprocessed_dense, y_encoded)

    transformation_dict = {
        "ohe_cols": ohe_cols,
        "word_index": word_index,
        "idf_dict": idf_dict,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "label_dict": label_dict,
    }

    return rf , transformation_dict

def prep_input(X, transformation_dict):

    cat_cols = transformation_dict["cat_cols"]
    ohe_cols = transformation_dict["ohe_cols"]
    word_index = transformation_dict["word_index"]
    idf_dict = transformation_dict["idf_dict"]
    numeric_cols = transformation_dict["numeric_cols"]

    text_cols = [
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?"
    ]
    label_col = "Label"

    for c in ["id", label_col]:
        if c in X.columns:
            X.drop(columns=[c], inplace=True, errors="ignore")

    X["combined_text"] = X[text_cols].apply(lambda row: " ".join(row.astype(str)), axis=1)
    X.drop(columns=text_cols, inplace=True)

    for cat_col in cat_cols:
        transform_one_hot_encode_inplace(X, cat_col, ohe_cols)

    text_series = X["combined_text"].fillna("").astype(str)
    X.drop(columns=["combined_text"], inplace=True)

    N = len(text_series)
    vocab_size = len(word_index)
    X_text = np.zeros((N, vocab_size), dtype=float)

    for i, text in enumerate(text_series):
        counts = {}
        for w in text.lower().split():
            if w in word_index:
                counts[w] = counts.get(w, 0) + 1

        for w, tf in counts.items():
            j = word_index[w]
            idf_val = idf_dict.get(w, 0.0)
            X_text[i, j] = tf * idf_val

    row_norms = np.sqrt(np.sum(X_text ** 2, axis=1, keepdims=True))
    nonzero_mask = row_norms[:, 0] != 0
    X_text[nonzero_mask] /= row_norms[nonzero_mask]

    X_ohe = X[ohe_cols].values if ohe_cols else np.zeros((N, 0))
    if all(c in X.columns for c in numeric_cols):
        X_numeric = X[numeric_cols].values
    else:
        X_numeric = np.zeros((N, 0))

    X_final = np.hstack([X_ohe, X_text, X_numeric]).astype(float)

    return X_final

def run_model(rf, X_final, transformation_dict):
    predictions = rf.predict(X_final)

    labels = transformation_dict["label_dict"]
    labels = {v: k for k, v in labels.items()}

    predictions_out = []

    for i in predictions:
            predictions_out.append(labels[i])

    return predictions_out

import numpy as np
import matplotlib.pyplot as plt

X, y = open_data("cleaned_data_combined.csv")

td = model_parms.td
rf_dict = model_parms.rf
rf = load_forest_from_python_dict(rf_dict)


accuracies = []

for i in range(100):
    X_train, X_test, y_train, y_test = data_split(X.copy(), y.copy())

    X_test_prepped = prep_input(X_test.copy(), td)

    output = run_model(rf, X_test_prepped, td)

    accuracy = np.mean(output == y_test)
    accuracies.append(accuracy)

    print(f"Run {i+1}: Accuracy = {accuracy:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, 101), accuracies, marker='o')
plt.title("Random Forest Accuracy over 10 Test Splits")
plt.xlabel("Run")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.show()