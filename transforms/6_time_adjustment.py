"""
Palantir Foundry PySpark Transform: Enrollment Date Correction (Simplified)

Purpose:
  Looks at each patient's REFILL row in the enriched matrix.
  - If refill = "Yes" at -12_months → shift ALL 3 rows forward by 12 months (2 buckets)
  - If refill = "Yes" at -6_months  → shift ALL 3 rows forward by 6 months (1 bucket)
  - If refill = "Yes" at BOTH       → shift by 12 months (use the largest)
  - Otherwise → no change

  Shifting means: move each time column's value forward by the shift amount,
  null out the vacated early buckets, and drop anything beyond 30 months.

  Rebuilds SUMMARY rows at the end.
"""
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from transforms.api import transform_df, Input, Output

MAX_MONTH_CUTOFF = 30


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    enriched_matrix=Input("ri.foundry.main.dataset.xxxxx"),
)
def compute(enriched_matrix):

    spark = enriched_matrix.sparkSession

    # ── Step 1: Separate data rows from SUMMARY rows ─────────────────────
    data_rows = enriched_matrix.filter(F.col("mrn") != "SUMMARY")

    # Identify time-bucket columns sorted numerically
    time_columns = sorted(
        [c for c in data_rows.columns if c.endswith("_months")],
        key=lambda x: int(x.replace("_months", ""))
    )
    bucket_values = [int(c.replace("_months", "")) for c in time_columns]
    bucket_to_col = {b: f"{b}_months" for b in bucket_values}

    # ── Step 2: Determine shift per patient from the REFILL row ──────────
    refill_rows = data_rows.filter(F.col("variable") == "refill")

    # Check if -12_months or -6_months has "Yes"
    has_minus_12 = F.col("-12_months").startswith("Yes")
    has_minus_6 = F.col("-6_months").startswith("Yes")

    patient_shift = refill_rows.select(
        "mrn",
        F.when(has_minus_12, F.lit(12))
         .when(has_minus_6, F.lit(6))
         .otherwise(F.lit(0))
         .alias("bucket_shift")
    )

    # Audit
    corrected = patient_shift.filter(F.col("bucket_shift") > 0).count()
    total = patient_shift.count()
    print(f"[Enrollment Correction] Patients shifted: {corrected}/{total}")

    # ── Step 3: Join shift onto ALL rows (all 3 rows per patient) ────────
    data_with_shift = data_rows.join(patient_shift, on="mrn", how="left").withColumn(
        "bucket_shift", F.coalesce(F.col("bucket_shift"), F.lit(0))
    )

    # ── Step 4: Shift time columns ───────────────────────────────────────
    #   For each bucket B: new value = old value at (B - shift)
    #   If (B - shift) doesn't exist → null
    #   If B > MAX_MONTH_CUTOFF after shift → null
    for b in bucket_values:
        col_name = bucket_to_col[b]

        # No shift → keep original
        expr = F.when(F.col("bucket_shift") == 0, F.col(col_name))

        # Shift by 6
        source_6 = b - 6
        if source_6 in bucket_to_col:
            expr = expr.when(F.col("bucket_shift") == 6, F.col(bucket_to_col[source_6]))
        else:
            expr = expr.when(F.col("bucket_shift") == 6, F.lit(None).cast(StringType()))

        # Shift by 12
        source_12 = b - 12
        if source_12 in bucket_to_col:
            expr = expr.when(F.col("bucket_shift") == 12, F.col(bucket_to_col[source_12]))
        else:
            expr = expr.when(F.col("bucket_shift") == 12, F.lit(None).cast(StringType()))

        # Fallback
        expr = expr.otherwise(F.lit(None).cast(StringType()))

        # Null out anything beyond cutoff for shifted patients
        if b > MAX_MONTH_CUTOFF:
            expr = F.when(F.col("bucket_shift") > 0, F.lit(None).cast(StringType())) \
                    .otherwise(F.col(col_name))

        data_with_shift = data_with_shift.withColumn(col_name, expr)

    # ── Step 5: Update enrollment_start_date for shifted patients ────────
    data_with_shift = data_with_shift.withColumn(
        "correct_enrollment_date",
        F.when(
            F.col("bucket_shift") > 0,
            F.expr("add_months(cast(enrollment_start_date as date), -bucket_shift)").cast("string")
        ).otherwise(F.col("enrollment_start_date"))
    ).withColumn(
        "enrollment_start_date",
        F.when(
            F.col("bucket_shift") > 0,
            F.col("correct_enrollment_date")
        ).otherwise(F.col("enrollment_start_date"))
    )

    # Drop helper
    data_corrected = data_with_shift.drop("bucket_shift")

    # ── Step 6: Rebuild SUMMARY rows ─────────────────────────────────────
    non_time_cols = [c for c in data_corrected.columns if c not in time_columns]
    final_cols = non_time_cols + time_columns

    def build_summary_row(df_var, variable_label, all_cols, t_cols):
        row = {}
        for c in all_cols:
            if c in t_cols:
                total = df_var.filter(F.col(c).isNotNull()).count()
                positive = df_var.filter(F.col(c).startswith("Yes")).count()
                row[c] = f"{positive}/{total}"
            elif c == "mrn":
                row[c] = "SUMMARY"
            elif c == "variable":
                row[c] = variable_label
            else:
                row[c] = None
        return row

    variables = [row["variable"] for row in data_corrected.select("variable").distinct().collect()]

    summary_rows = []
    for var_label in sorted(variables):
        var_df = data_corrected.filter(F.col("variable") == var_label)
        summary_rows.append(build_summary_row(var_df, var_label, final_cols, time_columns))

    summary_schema = StructType([StructField(c, StringType(), True) for c in final_cols])
    summary_df = spark.createDataFrame(summary_rows, summary_schema)

    # ── Step 7: Union, cast dates, sort ──────────────────────────────────
    for c in non_time_cols:
        if c in ("enrollment_start_date", "enrollment_end_date", "correct_enrollment_date"):
            data_corrected = data_corrected.withColumn(c, F.col(c).cast("string"))

    df_final = data_corrected.select(*final_cols).union(summary_df)

    df_final = df_final.orderBy(
        F.when(F.col("mrn") == "SUMMARY", 1).otherwise(0),
        "variable",
        "mrn",
    )

    return df_final