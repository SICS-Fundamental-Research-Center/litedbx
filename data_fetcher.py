import pandas as pd
import random
from tqdm import tqdm


def get_data(dataset: str, query: str, train_size: int):
    df = pd.read_csv(f"data/{dataset}_{query}.csv").reset_index(drop=True)
    X = df.drop(columns=["label"])
    Y = df["label"]
    return split_balanced(X, Y, train_size)


def split_basic(X: pd.DataFrame, Y: pd.Series, train_size: int):
    train_indices = X.sample(n=train_size, random_state=42).index
    test_indices = X.index.difference(train_indices)

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    Y_train = Y.loc[train_indices]
    Y_test = Y.loc[test_indices]

    return X_train, X_test, Y_train, Y_test


def split_balanced(X: pd.DataFrame, Y: pd.Series, train_size: int, lambda_prior: float = 5.0):
    random.seed(114514)

    D1 = set(X[X["LLM_label"] == 1].index)
    D0 = set(X[X["LLM_label"] == 0].index)

    p_hat = len(D1) / len(X)

    a = lambda_prior * p_hat
    b = lambda_prior * (1 - p_hat)

    S = set()
    na, nb = 0, 0

    for _ in range(train_size):
        pi_1 = a / (a + b)
        c = 1 if random.random() < pi_1 else 0

        if c == 1:
            available = D1 - S
        else:
            available = D0 - S

        x = random.choice(list(available))
        y = Y[x]

        a = a + y
        b = b + (1 - y)

        na += (y == 1)
        nb += (y == 0)

        S.add(x)

    train_indices = list(S)
    test_indices = X.index.difference(train_indices)

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    Y_train = Y.loc[train_indices]
    Y_test = Y.loc[test_indices]

    print(f"Class 1 in train: {na}, \tClass 0 in train: {nb}, \tRatio: {na/(na+nb):.4f}")
    print(f"Class 1 in real-labels: {Y.sum()}, \tClass 0 in real-labels: {len(Y) - Y.sum()}, \tRatio: {Y.sum()/len(Y):.4f}")
    print(f"Class 1 in pseudo-labels: {len(D1)}, \tClass 0 in pseudo-labels: {len(D0)}, \tRatio: {len(D1)/len(X):.4f}")

    return X_train, X_test, Y_train, Y_test



def split_coreset(X: pd.DataFrame, Y: pd.Series, train_size: int):
    n_bins = min(5, max(2, len(X) // 20))  # Adaptive bin count
    binned_df = pd.DataFrame(index=X.index)

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            try:
                binned_df[col] = pd.qcut(X[col], q=n_bins, duplicates='drop')
            except ValueError:
                binned_df[col] = pd.cut(X[col], bins=n_bins)
        else:
            binned_df[col] = X[col].astype(str)

    remaining_indices = set(X.index)
    train_indices = []
    covered_bins = set()

    for _ in tqdm(range(train_size), desc="Selecting coreset"):
        best_idx = None
        best_new_bins = 0

        for idx in remaining_indices:
            # Create bin signature for this row
            bin_signature = tuple(str(binned_df.loc[idx, col]) for col in binned_df.columns)

            # Count how many bins this row would add that we haven't covered
            new_bins = 0
            for col_idx, bin_val in enumerate(bin_signature):
                bin_key = (col_idx, bin_val)
                if bin_key not in covered_bins:
                    new_bins += 1

            # Track the row that adds the most new bins
            if new_bins > best_new_bins:
                best_new_bins = new_bins
                best_idx = idx

        if best_idx is not None:
            train_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            # Update covered bins
            bin_signature = tuple(str(binned_df.loc[best_idx, col]) for col in binned_df.columns)
            for col_idx, bin_val in enumerate(bin_signature):
                covered_bins.add((col_idx, bin_val))
        else:
            # All bins covered, randomly sample from remaining
            if remaining_indices:
                import random
                best_idx = random.choice(list(remaining_indices))
                train_indices.append(best_idx)
                remaining_indices.remove(best_idx)

    # Split the data
    test_indices = X.index.difference(train_indices)

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    Y_train = Y.loc[train_indices]
    Y_test = Y.loc[test_indices]

    return X_train, X_test, Y_train, Y_test



if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = get_data("medical", "q1", 50)
    print(Y_train.tolist())
