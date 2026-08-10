# sql_toolkit.py
"""
WHAT THIS DOES:
Loads the Adult Income data into a real SQLite database, and
provides a place to run SQL queries against it — treating SQL
as a genuine feature engineering tool, not just a syntax exercise.
"""
import sqlite3
import pandas as pd


def create_database(df: pd.DataFrame, db_path: str = "adult_income.db",
                     table_name: str = "adults") -> sqlite3.Connection:
    """
    Load a DataFrame into a real SQLite database file. Returns an
    open connection you can query against with plain SQL.
    """
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=True, index_label="id")
    print(f"Loaded {len(df)} rows into '{table_name}' table in {db_path}")
    return conn


def create_country_region_table(conn: sqlite3.Connection):
    """
    A small companion lookup table — deliberately built to give
    today's JOIN something real to demonstrate on, mapping country
    to a broader region (a genuinely useful engineered feature:
    a model may generalize better from 6 regions than 40
    countries, many with very few observations each).
    """
    country_region = {
        "United-States": "North America", "Canada": "North America",
        "Mexico": "North America",
        "England": "Europe", "Germany": "Europe", "France": "Europe",
        "Italy": "Europe", "Poland": "Europe", "Portugal": "Europe",
        "India": "Asia", "China": "Asia", "Japan": "Asia",
        "Philippines": "Asia", "Vietnam": "Asia", "South": "Asia",
        "Cuba": "Latin America", "Jamaica": "Latin America",
        "Puerto-Rico": "Latin America", "Honduras": "Latin America",
        "Unknown": "Unknown",
    }
    rows = [{"country": k, "region": v} for k, v in country_region.items()]
    lookup_df = pd.DataFrame(rows)

    lookup_df.to_sql("country_region", conn, if_exists="replace", index=False)
    print(f"Created country_region lookup table with {len(lookup_df)} mapped countries "
          f"(unmapped countries will show as NULL after a LEFT JOIN — a real, "
          f"honest gap worth seeing rather than hiding)")


def run_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Run any SQL query, return results as a DataFrame — the bridge back to pandas."""
    return pd.read_sql_query(query, conn)