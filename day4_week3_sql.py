# day4_week3_sql.py
from data_loader import load_adult_income
from preprocessing import clean_columns, handle_missing
from sql_toolkit import create_database, create_country_region_table, run_query

df = load_adult_income()
df = clean_columns(df)
df = handle_missing(df)

conn = create_database(df)
create_country_region_table(conn)

# ------------------------------------------------------------------
# PART 1: Basics — SELECT, WHERE, ORDER BY, LIMIT
# ------------------------------------------------------------------
print("=" * 60)
print("PART 1: BASIC FILTERING")
print("=" * 60)
result = run_query(conn, """
    SELECT age, occupation, "hours-per-week", income
    FROM adults
    WHERE age > 50 AND income = '>50K'
    ORDER BY "hours-per-week" DESC
    LIMIT 5
""")
print(result)

# ------------------------------------------------------------------
# PART 2: GROUP BY + aggregates — direct SQL equivalent of pandas
# groupby, to confirm both tools genuinely agree
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: GROUP BY — average hours worked per occupation")
print("=" * 60)
result = run_query(conn, """
    SELECT occupation,
           COUNT(*) AS num_people,
           ROUND(AVG("hours-per-week"), 1) AS avg_hours,
           ROUND(AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END), 3) AS high_income_rate
    FROM adults
    GROUP BY occupation
    ORDER BY high_income_rate DESC
""")
print(result)

# ------------------------------------------------------------------
# PART 3: JOIN — enrich with the country_region lookup table
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3: LEFT JOIN — mapping country to region")
print("=" * 60)
result = run_query(conn, """
    SELECT a.id, a."native-country", cr.region
    FROM adults a
    LEFT JOIN country_region cr ON a."native-country" = cr.country
    LIMIT 10
""")
print(result)

unmapped = run_query(conn, """
    SELECT a."native-country", COUNT(*) AS n
    FROM adults a
    LEFT JOIN country_region cr ON a."native-country" = cr.country
    WHERE cr.region IS NULL
    GROUP BY a."native-country"
    ORDER BY n DESC
""")
print(f"\nUnmapped countries (would need adding to the lookup table for production use):")
print(unmapped)

# ------------------------------------------------------------------
# PART 4: Window functions — GROUP BY without collapsing rows
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4: WINDOW FUNCTIONS — per-row comparison to group average")
print("=" * 60)
result = run_query(conn, """
    SELECT id, occupation, "hours-per-week",
           ROUND(AVG("hours-per-week") OVER (PARTITION BY occupation), 1) AS avg_hours_in_occupation,
           ROUND("hours-per-week" - AVG("hours-per-week") OVER (PARTITION BY occupation), 1) AS hours_vs_peers
    FROM adults
    LIMIT 10
""")
print(result)
print(f"\nNote: every original row is preserved (still {len(result)} rows shown), "
      f"unlike Part 2's GROUP BY which collapsed to one row per occupation")

# ------------------------------------------------------------------
# PART 5: Turn a window function into an actual engineered feature,
# pulled back into pandas for use in tomorrow/Week 5's modeling
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 5: SQL-ENGINEERED FEATURE — hours relative to occupation peers")
print("=" * 60)
engineered = run_query(conn, """
    SELECT id,
           "hours-per-week" - AVG("hours-per-week") OVER (PARTITION BY occupation) AS hours_vs_occupation_avg,
           RANK() OVER (PARTITION BY "education_num" ORDER BY "capital-gain" DESC) AS capital_gain_rank_in_education
    FROM adults
""")
print(engineered.describe())
engineered.to_csv("sql_engineered_features.csv", index=False)
print("\nSaved sql_engineered_features.csv — ready to merge back with the "
      "Wednesday pipeline output by 'id' when needed")

conn.close()