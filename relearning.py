from concurrent.futures import ProcessPoolExecutor, as_completed
from script import (
    catch_exceptions,
    create_time_series,
    fit_exp_forgetting_curve,
    fit_power_forgetting_curve,
    fit_exp_forgetting_curve_with_intercept,
    exp_forgetting_curve,
    power_forgetting_curve,
    exp_forgetting_curve_with_intercept,
)
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss  # type: ignore
from tqdm import tqdm

PLOT = False
DATA_PATH = Path("../anki-revlogs-10k")


@catch_exceptions
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
    df = dataset[
        (dataset["t_history"] == "0,0,0,1") & (dataset["r_history"] == "1,3,3,3")
    ]
    if df.empty:
        return None
    most_common_deck_id = df["deck_id"].value_counts().idxmax()
    df = df[df["deck_id"] == most_common_deck_id]
    if df["elapsed_days"].unique().size < 3 or df["y"].mean() > 0.99 or len(df) < 100:
        return None
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
    exp_with_intercept_params = fit_exp_forgetting_curve_with_intercept(grouped)
    exp_loss = log_loss(
        df["y"], exp_forgetting_curve(df["elapsed_days"], exp_params), labels=[0, 1]
    )
    power_loss = log_loss(
        df["y"],
        power_forgetting_curve(df["elapsed_days"], *power_params),
        labels=[0, 1],
    )
    exp_with_intercept_loss = log_loss(
        df["y"],
        exp_forgetting_curve_with_intercept(
            df["elapsed_days"], *exp_with_intercept_params
        ),
        labels=[0, 1],
    )
    t_start = grouped["elapsed_days"].min()
    t_end = grouped["elapsed_days"].max()
    t_span = t_end - t_start
    retention = df["y"].mean()

    results = {
        "user_id": user_id,
        "sample_size": df.shape[0],
        "stability_exp": exp_params,
        "loss_exp": exp_loss,
        "stability_pow": power_params[0],
        "decay_pow": power_params[1],
        "loss_pow": power_loss,
        "stability_exp_with_intercept": exp_with_intercept_params[0],
        "intercept_exp_with_intercept": exp_with_intercept_params[1],
        "loss_exp_with_intercept": exp_with_intercept_loss,
        "t_span": int(t_span),
        "retention": retention,
    }

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
        plt.plot(
            t_range,
            exp_forgetting_curve_with_intercept(t_range, *exp_with_intercept_params),
            label=f"Exp with intercept s:{exp_with_intercept_params[0]:.2f} intercept:{exp_with_intercept_params[1]:.2f} loss:{exp_with_intercept_loss:.4f}",
        )
        plt.title(f"User ID: {user_id}, Sample Size: {df.shape[0]}")
        plt.legend()
        Path("relearning_plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"relearning_plots/{user_id}.png")
        plt.close()

    return results


if __name__ == "__main__":
    all_results = []
    user_ids = range(1, 1001)

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_user = {
            executor.submit(fit_forgetting_curve, user_id): user_id
            for user_id in user_ids
        }

        total_users = len(user_ids)
        processed_users = 0

        with tqdm(total=total_users, desc="Processing users") as progress_bar:
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                results_df, error = future.result()
                if error is None:
                    if results_df is not None:
                        all_results.append(results_df)
                else:
                    print(f"Error processing user {user_id}: {error}")

                processed_users += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {"Completed": f"{processed_users}/{total_users}"}
                )

    final_results = pd.DataFrame(all_results)
    final_results.sort_values(by=["user_id"], inplace=True)
    final_results.to_csv("relearning_results.csv", index=False)
