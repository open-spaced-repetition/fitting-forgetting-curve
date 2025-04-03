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
from concurrent.futures import ProcessPoolExecutor, as_completed

max_seq_len: int = 64
DATA_PATH = Path("../anki-revlogs-10k")
PLOT = True


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
    columns = ["card_id", "rating", "elapsed_days"]
    df_revlogs = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)], columns=columns
    )
    dataset = create_time_series(df_revlogs)
    df_cards = pd.read_parquet(DATA_PATH / "cards", filters=[("user_id", "=", user_id)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(DATA_PATH / "decks", filters=[("user_id", "=", user_id)])
    df_decks.drop(columns=["user_id"], inplace=True)
    dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
        df_decks, on="deck_id", how="left"
    )
    dataset.dropna(inplace=True)

    pretrainset = dataset[dataset["i"] == 2]
    first_ratings = pretrainset["first_rating"].unique()

    results = []
    for first_rating in first_ratings:
        df = pretrainset[pretrainset["first_rating"] == first_rating]
        most_common_deck_id = df["deck_id"].value_counts().idxmax()
        df = df[df["deck_id"] == most_common_deck_id]
        most_common_r_history = df["r_history"].value_counts().idxmax()
        df = df[df["r_history"] == most_common_r_history]
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

        exp_loss = log_loss(
            df["y"], exp_forgetting_curve(df["elapsed_days"], exp_params), labels=[0, 1]
        )
        power_loss = log_loss(
            df["y"],
            power_forgetting_curve(df["elapsed_days"], *power_params),
            labels=[0, 1],
        )
        t_start = grouped["elapsed_days"].min()
        t_end = grouped["elapsed_days"].max()
        t_span = t_end - t_start

        results.append(
            {
                "user_id": user_id,
                "first_rating": first_rating,
                "sample_size": df.shape[0],
                "stability_exp": exp_params,
                "loss_exp": exp_loss,
                "stability_pow": power_params[0],
                "decay_pow": power_params[1],
                "loss_pow": power_loss,
                "t_span": int(t_span),
            }
        )

        if PLOT:
            t_range = np.linspace(0, t_end, 100)
            plt.scatter(
                grouped["elapsed_days"],
                grouped["retention"],
                grouped["total_cnt"] / grouped["total_cnt"].sum() * 100,
                label="Actual",
            )
            plt.plot(
                t_range,
                exp_forgetting_curve(t_range, exp_params),
                label=f"Exp s:{exp_params:.2f} loss:{exp_loss:.4}",
            )
            plt.plot(
                t_range,
                power_forgetting_curve(t_range, *power_params),
                label=f"Pow s:{power_params[0]:.2f} d:{power_params[1]:.2f} loss:{power_loss:.4f}",
            )
            plt.title(
                f"User ID: {user_id}, First Rating: {first_rating}, Sample Size: {df.shape[0]}"
            )
            plt.legend()
            plt.savefig(f"plots/{user_id}_{first_rating}.png")
            plt.close()

    return pd.DataFrame(results)


if __name__ == "__main__":
    all_results = []
    user_ids = range(1, 101)

    with ProcessPoolExecutor() as executor:
        future_to_user = {
            executor.submit(fit_forgetting_curve, user_id): user_id
            for user_id in user_ids
        }

        # 收集结果
        for future in as_completed(future_to_user):
            user_id = future_to_user[future]
            try:
                results_df = future.result()
                all_results.append(results_df)
            except Exception as e:
                print(f"Error processing user {user_id}: {str(e)}")

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.sort_values(by=["user_id", "first_rating"], inplace=True)
    final_results.to_csv("fitting_results.csv", index=False)
