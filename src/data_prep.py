from pathlib import Path
import pandas as pd
import duckdb

RAW = Path("data/raw/online_retail_II.xlsx")
PARQUET = Path("data/processed/retail_raw.parquet")
DB = Path("data/processed/retail.duckdb")


def load_raw() -> pd.DataFrame:
    # Read invoice and stock code as text, since cancellations start with "C"
    # and some stock codes contain letters
    return pd.read_excel(RAW, dtype={"InvoiceNo": str, "StockCode": str})


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "InvoiceNo": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "UnitPrice": "price",
            "CustomerID": "customer_id",
            "Country": "country",
        }
    )
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype("Int64")  # keeps nulls
    df["description"] = df["description"].astype("string")
    df["country"] = df["country"].astype("string")
    return df


def main():
    df = clean_columns(load_raw())

    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET, index=False)

    con = duckdb.connect(str(DB))
    con.execute(
        "CREATE OR REPLACE TABLE raw_transactions AS "
        "SELECT * FROM read_parquet(?)",
        [str(PARQUET)],
    )
    print(
        con.execute(
            "SELECT COUNT(*) AS n_rows, "
            "MIN(invoice_date) AS first_date, "
            "MAX(invoice_date) AS last_date, "
            "COUNT(DISTINCT customer_id) AS customers "
            "FROM raw_transactions"
        ).fetchall()
    )
    con.close()


if __name__ == "__main__":
    main()