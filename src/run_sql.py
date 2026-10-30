import sys
from pathlib import Path
import duckdb

DB = Path("data/processed/retail.duckdb")


def main():
    sql_file = Path(sys.argv[1])
    table = sys.argv[2] if len(sys.argv) > 2 else "clean_transactions"

    con = duckdb.connect(str(DB))
    con.execute(sql_file.read_text())
    print(f"Ran {sql_file}")

    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n} rows")
    print(con.execute(f"SELECT * FROM {table} LIMIT 5").df())
    con.close()


if __name__ == "__main__":
    main()