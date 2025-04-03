import numpy as np
import pandas as pd
from pathlib import Path
from itertools import accumulate
from fsrs_optimizer import (  # type: ignore
    remove_outliers,
    remove_non_continuous_rows,
)
from scipy.optimize import minimize  # type: ignore
from sklearn.metrics import log_loss  # type: ignore
import matplotlib.pyplot as plt


max_seq_len: int = 64
DATA_PATH = Path("../anki-revlogs-10k")


def cum_concat(x):
    return list(accumulate(x))


def create_time_series(df):
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.drop(df[df["i"] > max_seq_len * 2].index, inplace=True)
    card_id_to_first_rating = df.groupby("card_id")["rating"].first().to_dict()
    t_history_list = df.groupby("card_id", group_keys=False)["elapsed_days"].apply(
        lambda x: cum_concat([[max(0, i)] for i in x])
    )
    r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history_list for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history_list for item in sublist
    ]
    last_rating = []
    for t_sublist, r_sublist in zip(t_history_list, r_history_list):
        for t_history, r_history in zip(t_sublist, r_sublist):
            flag = True
            for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                if t > 0:
                    last_rating.append(r)
                    flag = False
                    break
            if flag:
                last_rating.append(r_history[0])
    df["last_rating"] = last_rating
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df.drop(df[df["elapsed_days"] == 0].index, inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["first_rating"] = df["card_id"].map(card_id_to_first_rating).astype(str)
    df["delta_t"] = df["elapsed_days"]
    filtered_dataset = (
        df[df["i"] == 2]
        .groupby(by=["first_rating"], as_index=False, group_keys=False)[df.columns]
        .apply(remove_outliers)
    )
    if filtered_dataset.empty:
        return pd.DataFrame()
    df[df["i"] == 2] = filtered_dataset
    df.dropna(inplace=True)
    df = df.groupby("card_id", as_index=False, group_keys=False)[df.columns].apply(
        remove_non_continuous_rows
    )
    return df[df["elapsed_days"] > 0].sort_values(by=["review_th"])


def exp_forgetting_curve(t, s):
    return 0.9 ** (t / s)


def power_forgetting_curve(t, s, decay):
    factor = 0.9 ** (1 / decay) - 1
    return (1 + factor * t / s) ** decay


def fit_exp_forgetting_curve(df):
    x = df["elapsed_days"].values
    y = df["retention"].values
    size = df["total_cnt"].values

    def loss(params):
        s = params
        y_pred = exp_forgetting_curve(x, s)
        loss = sum(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) * size)
        return loss

    res = minimize(loss, x0=1, bounds=[(0.1, 36500)])
    return res.x[0]


def fit_power_forgetting_curve(df):
    x = df["elapsed_days"].values
    y = df["retention"].values
    size = df["total_cnt"].values

    def loss(params):
        s, decay = params
        y_pred = power_forgetting_curve(x, s, decay)
        loss = sum(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) * size)
        return loss

    res = minimize(loss, x0=(1, -0.5), bounds=((0.1, 36500), (-1, -0.1)))
    return res.x


def fit_forgetting_curve(user_id: int):
    print(f"User ID: {user_id}")
    df_revlogs = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_time_series(df_revlogs)

    pretrainset = dataset[dataset["i"] == 2]

    first_ratings = pretrainset["first_rating"].unique()

    for first_rating in first_ratings:
        df = pretrainset[pretrainset["first_rating"] == first_rating]
        grouped = (
            df.groupby(by=["elapsed_days"], group_keys=False)
            .agg(
                retention=("y", "mean"),
                total_cnt=("y", "count"),
            )
            .reset_index()
        )
        exp_params = fit_exp_forgetting_curve(grouped)
        power_params = fit_power_forgetting_curve(grouped)

        print(f"First Rating: {first_rating}")
        print(f"Number of samples: {df.shape[0]}")
        print(f"Exponential model parameters: {exp_params}")
        print(f"Power model parameters: {power_params}")

        exp_loss = log_loss(
            df["y"], exp_forgetting_curve(df["elapsed_days"], exp_params), labels=[0, 1]
        )
        power_loss = log_loss(
            df["y"],
            power_forgetting_curve(df["elapsed_days"], *power_params),
            labels=[0, 1],
        )
        print(f"Exponential model log loss: {exp_loss:.4f}")
        print(f"Power model log loss: {power_loss:.4f}")
        print("-" * 50)

        plt.scatter(
            grouped["elapsed_days"],
            grouped["retention"],
            grouped["total_cnt"] / grouped["total_cnt"].sum() * 100,
            label="Actual",
        )
        plt.plot(
            grouped["elapsed_days"],
            exp_forgetting_curve(grouped["elapsed_days"], exp_params),
            label="Exponential",
        )
        plt.plot(
            grouped["elapsed_days"],
            power_forgetting_curve(grouped["elapsed_days"], *power_params),
            label="Power",
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    for user_id in range(1, 10):
        fit_forgetting_curve(user_id)
