"""
Palantir Foundry PySpark Transform: EPIC Medication Safety Check + Provider Enrichment
(Parallel-Row Approach)

Purpose:
  For each patient, produce THREE rows in the timepoint matrix:
    1. variable = "diarrhea"  → existing Yes/No diarrhea status per timepoint
    2. variable = "epic_med"  → Yes/No whether EPIC medication records cover that timepoint
    3. variable = "refill"    → Yes/No whether EPIC refill records cover that timepoint

  Plus a flat `note_author` column (provider), and SUMMARY rows per variable.

Input:
  - The prior-diarrhea-enriched pivot table (output of the previous transform)
  - EPIC administered medications
  - EPIC order medications
  - Note providers table

Output:
  Same RID — replaces the previous version with the richer parallel-row layout.
"""
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType
)
from transforms.api import transform_df, Input, Output

# ── Regex pattern to catch Daybue / trofinetide variations ───────────────────
DAYBUE_PATTERN = r"(?i)(daybue|trofinetide|trofine|trofi\s*netide)"


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    final_matrix=Input("ri.foundry.main.dataset.xxxxx"),
    administered_meds=Input("ri.foundry.main.dataset.xxxxx"),
    order_meds=Input("ri.foundry.main.dataset.xxxxx"),
    note_providers=Input("ri.foundry.main.dataset.xxxxx"),
)
def compute(final_matrix, administered_meds, order_meds, note_providers):

    # ========================================================================
    # STEP 1: Separate data rows from SUMMARY row; identify time columns
    # ========================================================================

    patient_matrix = final_matrix.filter(F.col("mrn") != "SUMMARY")
    # Drop old SUMMARY — we rebuild it at the end for all three variables

    # Identify time-bucket columns (e.g. -12_months, -6_months, 0_months, …)
    time_columns = sorted(
        [c for c in patient_matrix.columns if c.endswith("_months")],
        key=lambda x: int(x.replace("_months", ""))
    )

    meta_cols = ["mrn", "enrollment_start_date", "enrollment_end_date"]

    # Collect MRN list for push-down filtering on large EPIC tables
    mrn_list = [row["mrn"] for row in patient_matrix.select("mrn").distinct().collect()]

    # ========================================================================
    # STEP 2: Build note_author lookup from providers table
    # ========================================================================

    providers_lookup = note_providers.select(
        F.col("MRN").cast(StringType()).alias("mrn"),
        F.col("Provider").alias("provider_name"),
    ).filter(
        F.col("mrn").isin(mrn_list) &
        F.col("provider_name").isNotNull()
    )

    providers_agg = providers_lookup.groupBy("mrn").agg(
        F.concat_ws(", ",
            F.array_sort(F.collect_set("provider_name"))
        ).alias("note_author")
    )

    # ========================================================================
    # STEP 3: Filter EPIC Administered Medications → per-patient date list
    # ========================================================================

    admin_filtered = administered_meds.filter(
        F.col("mrn").isin(mrn_list)
    ).filter(
        F.regexp_like(F.col("simple_generic"), F.lit(DAYBUE_PATTERN)) |
        F.regexp_like(F.col("display_name"),   F.lit(DAYBUE_PATTERN)) |
        F.regexp_like(F.col("description"),    F.lit(DAYBUE_PATTERN))
    ).select(
        F.col("mrn"),
        F.to_date(F.col("taken_time")).alias("event_date"),
        F.lit("epic_med").alias("source"),
    )

    # ========================================================================
    # STEP 4: Filter EPIC Order Medications → per-patient date ranges
    # For epic_med: use order_start_date and order_end_date to define coverage
    # For refill:   same rows but only where refills > 0
    # ========================================================================

    orders_raw = order_meds.filter(
        F.col("mrn").isin(mrn_list)
    ).filter(
        F.regexp_like(F.col("simple_generic"), F.lit(DAYBUE_PATTERN)) |
        F.regexp_like(F.col("display_name"),   F.lit(DAYBUE_PATTERN)) |
        F.regexp_like(F.col("description"),    F.lit(DAYBUE_PATTERN))
    ).select(
        F.col("mrn"),
        F.to_date(F.col("order_start_date")).alias("order_start"),
        F.to_date(F.col("order_end_date")).alias("order_end"),
        F.when(
            F.col("refills").isNotNull() & (F.col("refills").cast("int") > 0),
            F.lit(True)
        ).otherwise(F.lit(False)).alias("has_refill"),
    )

    # ========================================================================
    # STEP 5: For each patient, determine which time buckets have EPIC med
    #         coverage and which have refill coverage.
    #
    # Approach: unpivot the patient_matrix to get (mrn, enrollment_start_date,
    # time_bucket) triples, compute the calendar window for each bucket,
    # then check overlap against EPIC records.
    # ========================================================================

    # 5a. Get enrollment dates per patient (cast back to date)
    patient_dates = patient_matrix.select(
        "mrn",
        F.col("enrollment_start_date").cast("date").alias("enrollment_start_date"),
    )

    # 5b. Create a row per (mrn, time_bucket) with the calendar window
    #     For bucket B (in months): window = [start + B months, start + (B+6) months)
    bucket_values = [int(c.replace("_months", "")) for c in time_columns]

    # Build a small DataFrame of bucket definitions
    spark = patient_matrix.sparkSession
    bucket_rows = [(b, f"{b}_months") for b in bucket_values]
    bucket_df = spark.createDataFrame(bucket_rows, ["bucket_month", "time_bucket_label"])

    # Cross join patients × buckets to get every (patient, bucket) pair
    patient_buckets = patient_dates.crossJoin(bucket_df)

    # Compute the calendar start/end of each bucket
    patient_buckets = patient_buckets.withColumn(
        "bucket_start",
        F.expr("add_months(enrollment_start_date, bucket_month)")
    ).withColumn(
        "bucket_end",
        F.expr("add_months(enrollment_start_date, bucket_month + 6)")
    )

    # 5c. Check EPIC administered med overlap:
    #     An admin record overlaps a bucket if event_date falls within [bucket_start, bucket_end)
    admin_overlap = patient_buckets.join(admin_filtered, on="mrn", how="inner").filter(
        (F.col("event_date") >= F.col("bucket_start")) &
        (F.col("event_date") < F.col("bucket_end"))
    ).select("mrn", "time_bucket_label").distinct().withColumn(
        "admin_hit", F.lit(True)
    )

    # 5d. Check EPIC order med overlap:
    #     An order overlaps a bucket if [order_start, order_end] intersects [bucket_start, bucket_end)
    #     Intersection condition: order_start < bucket_end AND order_end >= bucket_start
    #     (order_end can be null → treat as open-ended / still active)
    order_overlap = patient_buckets.join(orders_raw, on="mrn", how="inner").filter(
        (F.col("order_start") < F.col("bucket_end")) &
        (
            F.col("order_end").isNull() |
            (F.col("order_end") >= F.col("bucket_start"))
        )
    )

    # epic_med: any order or admin overlaps
    order_med_overlap = order_overlap.select(
        "mrn", "time_bucket_label"
    ).distinct().withColumn("order_hit", F.lit(True))

    # refill: only orders with has_refill = True that overlap
    order_refill_overlap = order_overlap.filter(
        F.col("has_refill") == True
    ).select(
        "mrn", "time_bucket_label"
    ).distinct().withColumn("refill_hit", F.lit(True))

    # 5e. Combine: epic_med = admin_hit OR order_hit; refill = refill_hit
    all_buckets = patient_buckets.select("mrn", "time_bucket_label").distinct()

    epic_med_flags = all_buckets \
        .join(admin_overlap, on=["mrn", "time_bucket_label"], how="left") \
        .join(order_med_overlap, on=["mrn", "time_bucket_label"], how="left") \
        .withColumn(
            "epic_med_status",
            F.when(
                F.col("admin_hit").isNotNull() | F.col("order_hit").isNotNull(),
                F.lit("Yes")
            ).otherwise(F.lit("No"))
        ).select("mrn", "time_bucket_label", "epic_med_status")

    refill_flags = all_buckets \
        .join(order_refill_overlap, on=["mrn", "time_bucket_label"], how="left") \
        .withColumn(
            "refill_status",
            F.when(
                F.col("refill_hit").isNotNull(),
                F.lit("Yes")
            ).otherwise(F.lit("No"))
        ).select("mrn", "time_bucket_label", "refill_status")

    # ========================================================================
    # STEP 6: Pivot epic_med and refill flags into patient × timepoint matrices
    # ========================================================================

    epic_med_pivot = epic_med_flags.groupBy("mrn").pivot(
        "time_bucket_label", time_columns
    ).agg(F.first("epic_med_status"))

    refill_pivot = refill_flags.groupBy("mrn").pivot(
        "time_bucket_label", time_columns
    ).agg(F.first("refill_status"))

    # ========================================================================
    # STEP 7: Build the three parallel-row DataFrames
    #
    # Row 1 (diarrhea): existing patient_matrix rows as-is
    # Row 2 (epic_med): epic_med_pivot with same meta columns
    # Row 3 (refill):   refill_pivot with same meta columns
    #
    # All three share: mrn, enrollment_start_date, enrollment_end_date,
    #                  note_author, variable, <time_columns>
    # ========================================================================

    # 7a. Diarrhea rows — add variable label and provider
    diarrhea_rows = patient_matrix.withColumn("variable", F.lit("diarrhea"))
    diarrhea_rows = diarrhea_rows.join(providers_agg, on="mrn", how="left").withColumn(
        "note_author",
        F.when(F.col("note_author").isNull(), F.lit("No Provider Record"))
         .otherwise(F.col("note_author"))
    )

    # 7b. Epic med rows — attach enrollment dates and provider
    epic_med_rows = epic_med_pivot.join(
        patient_matrix.select("mrn", "enrollment_start_date", "enrollment_end_date"),
        on="mrn", how="left"
    ).withColumn("variable", F.lit("epic_med"))
    epic_med_rows = epic_med_rows.join(providers_agg, on="mrn", how="left").withColumn(
        "note_author",
        F.when(F.col("note_author").isNull(), F.lit("No Provider Record"))
         .otherwise(F.col("note_author"))
    )

    # 7c. Refill rows — attach enrollment dates and provider
    refill_rows = refill_pivot.join(
        patient_matrix.select("mrn", "enrollment_start_date", "enrollment_end_date"),
        on="mrn", how="left"
    ).withColumn("variable", F.lit("refill"))
    refill_rows = refill_rows.join(providers_agg, on="mrn", how="left").withColumn(
        "note_author",
        F.when(F.col("note_author").isNull(), F.lit("No Provider Record"))
         .otherwise(F.col("note_author"))
    )

    # ========================================================================
    # STEP 8: Standardise column order and union all three
    # ========================================================================

    output_cols = meta_cols + ["variable", "note_author"] + time_columns

    all_data = (
        diarrhea_rows.select(*output_cols)
        .union(epic_med_rows.select(*output_cols))
        .union(refill_rows.select(*output_cols))
    )

    # ========================================================================
    # STEP 9: Build SUMMARY rows — one per variable
    # ========================================================================

    def build_summary_row(df_var, variable_label):
        """Count positive/total per timepoint for a given variable subset."""
        row = {
            "mrn": "SUMMARY",
            "enrollment_start_date": None,
            "enrollment_end_date": None,
            "variable": variable_label,
            "note_author": None,
        }
        for col_name in time_columns:
            total    = df_var.filter(F.col(col_name).isNotNull()).count()
            positive = df_var.filter(F.col(col_name).startswith("Yes")).count()
            row[col_name] = f"{positive}/{total}"
        return row

    summary_diarrhea = build_summary_row(
        all_data.filter(F.col("variable") == "diarrhea"), "diarrhea"
    )
    summary_epic_med = build_summary_row(
        all_data.filter(F.col("variable") == "epic_med"), "epic_med"
    )
    summary_refill = build_summary_row(
        all_data.filter(F.col("variable") == "refill"), "refill"
    )

    summary_schema = StructType(
        [StructField(c, StringType(), True) for c in output_cols]
    )

    summary_df = spark.createDataFrame(
        [summary_diarrhea, summary_epic_med, summary_refill],
        summary_schema
    )

    # ========================================================================
    # STEP 10: Union data + summaries, cast dates to string, sort
    # ========================================================================

    df_final = all_data.select(
        "mrn",
        F.col("enrollment_start_date").cast("string"),
        F.col("enrollment_end_date").cast("string"),
        "variable",
        "note_author",
        *time_columns
    ).union(summary_df)

    # Sort: group by variable, then MRN within each variable, SUMMARY last
    df_final = df_final.orderBy(
        F.when(F.col("mrn") == "SUMMARY", 1).otherwise(0),
        "variable",
        "mrn",
    )

    return df_final