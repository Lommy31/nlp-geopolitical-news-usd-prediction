import os

import pandas as pd

INPUT_PATH = "data/raw/Informasi Kurs Jisdor.xlsx"
OUTPUT_PATH = "data/processed/usd_idr_jisdor_clean.csv"


def load_raw(path):
    df = pd.read_excel(path, header=4, usecols=["Tanggal", "Kurs"])
    df = df.rename(columns={"Tanggal": "Date", "Kurs": "USD_IDR"})
    return df


def clean(df):
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p")
    df = df.dropna(subset=["Date", "USD_IDR"])
    df = df[df["USD_IDR"] > 0]
    df = df.drop_duplicates(subset="Date")
    df = df.sort_values("Date").reset_index(drop=True)

    pct_change = df["USD_IDR"].pct_change().abs()
    spikes = df[pct_change > 0.05]
    for _, row in spikes.iterrows():
        print(f"warning: large day-over-day jump on {row['Date'].date()} ({row['USD_IDR']})")

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    df = load_raw(INPUT_PATH)
    df = clean(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")


if __name__ == "__main__":
    main()
