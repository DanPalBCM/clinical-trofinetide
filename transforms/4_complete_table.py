"""
Palantir Foundry PySpark Transform: Add Pre-Enrollment Diarrhea Columns (-6m, -12m)

Purpose:
  1. Take the existing patient × timepoint pivot table (0–30 months)
  2. Read prior phenotypes (keyed by patient_id) and map to mrn
  3. Parse prior_phenotypes JSON for Chronic/Recurrent Diarrhea
  4. Bucket observations into -6 month and -12 month windows relative to enrollment_start_date
  5. Append these two columns to the existing pivot table
  6. Rebuild the SUMMARY row to include the new columns
"""
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType
)
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    existing_pivot=Input("ri.foundry.main.dataset.xxxxx"),
    prior_phenotypes=Input("ri.foundry.main.dataset.xxxxx"),
    mrn_patient_id_mapping=Input("ri.foundry.main.dataset.xxxxx"),
)
def compute(existing_pivot, prior_phenotypes, mrn_patient_id_mapping):

    # ========================================================================
    # STEP 1: Load the existing pivot table and separate data rows vs SUMMARY
    # ========================================================================

    pivot_df = existing_pivot

    # Split out the SUMMARY row — we will rebuild it at the end
    data_rows = pivot_df.filter(F.col("mrn") != "SUMMARY")
    # Keep enrollment dates for bucketing later
    patient_meta = data_rows.select(
        "mrn",
        F.col("enrollment_start_date").cast("date").alias("enrollment_start_date"),
        F.col("enrollment_end_date").cast("date").alias("enrollment_end_date"),
    )

    # ========================================================================
    # STEP 2: Load the MRN ↔ patient_id mapping
    # ========================================================================

    mapping_df = mrn_patient_id_mapping.select(
        F.col("mrn"),
        F.col("patient_id"),
    ).distinct()

    # ========================================================================
    # STEP 3: Parse prior phenotypes JSON for diarrhea
    # ========================================================================

    diarrhea_schema = ArrayType(StructType([
        StructField("phenotype",            StringType(), True),
        StructField("present",              StringType(), True),
        StructField("chronicity_indicator", StringType(), True),
        StructField("severity_level",       StringType(), True),
        StructField("time",                 StringType(), True),
        StructField("time_description",     StringType(), True),
        StructField("evidence_text",        StringType(), True),
    ]))

    prior_df = prior_phenotypes

    # Clean JSON (strip markdown fences, normalise whitespace)
    prior_df = prior_df.withColumn(
        "pp_cleaned",
        F.when(
            F.col("prior_phenotypes").contains("```json"),
            F.regexp_extract(F.col("prior_phenotypes"), r"```json\s*([\s\S]*?)\s*```", 1)
        ).otherwise(F.col("prior_phenotypes"))
    ).withColumn(
        "pp_cleaned",
        F.regexp_replace(
            F.regexp_replace(F.col("pp_cleaned"), r"\n", " "),
            r"\s+", " "
        )
    ).withColumn(
        "pp_parsed",
        F.from_json(F.col("pp_cleaned"), diarrhea_schema)
    )

    # Filter to diarrhea entries (match both "Diarrhea" and "Chronic/Recurrent Diarrhea")
    prior_df = prior_df.withColumn(
        "diarrhea_entry",
        F.expr(
            "filter(pp_parsed, x -> lower(x.phenotype) like '%diarrhea%')"
        )
    ).withColumn(
        "diarrhea_entry", F.col("diarrhea_entry")[0]
    ).withColumn(
        "diarrhea_present",
        F.coalesce(F.col("diarrhea_entry.present"), F.lit("No"))
    ).withColumn(
        "severity_level",
        F.col("diarrhea_entry.severity_level")
    )

    # Resolve observation time: use extracted time if available, else contact_date
    prior_df = prior_df.withColumn(
        "phenotype_time_parsed",
        F.when(
            F.col("diarrhea_entry.time").isNotNull() & (F.col("diarrhea_entry.time") != ""),
            F.to_date(F.col("diarrhea_entry.time"), "M/d/yy")
        ).otherwise(F.lit(None).cast("date"))
    ).withColumn(
        "note_time",
        F.coalesce(F.col("phenotype_time_parsed"), F.col("contact_date"))
    )

    # ========================================================================
    # STEP 4: Map patient_id → mrn
    # ========================================================================

    prior_df = prior_df.join(mapping_df, on="patient_id", how="inner")

    # ========================================================================
    # STEP 5: Join with enrollment metadata to get enrollment_start_date
    # ========================================================================

    prior_df = prior_df.join(patient_meta, on="mrn", how="inner")

    # ========================================================================
    # STEP 6: Compute months from start and bucket into -6m / -12m windows
    #
    #   -12_months bucket: observations from -12 to -6 months before start
    #   -6_months  bucket: observations from  -6 to  0 months before start
    #
    # These are PRE-enrollment, so note_time < enrollment_start_date
    # ========================================================================

    prior_df = prior_df.filter(
        F.col("note_time").isNotNull() &
        (F.col("note_time") < F.col("enrollment_start_date"))
    ).withColumn(
        "months_from_start",
        F.months_between(F.col("note_time"), F.col("enrollment_start_date"))
    )

    # -6_months bucket: [-6, 0)  → months_from_start >= -6 AND < 0
    # -12_months bucket: [-12, -6) → months_from_start >= -12 AND < -6
    prior_df = prior_df.withColumn(
        "time_bucket_label",
        F.when(
            (F.col("months_from_start") >= -6) & (F.col("months_from_start") < 0),
            F.lit("-6_months")
        ).when(
            (F.col("months_from_start") >= -12) & (F.col("months_from_start") < -6),
            F.lit("-12_months")
        ).otherwise(F.lit(None))
    ).filter(
        F.col("time_bucket_label").isNotNull()
    )

    # ========================================================================
    # STEP 7: Aggregate by patient × time bucket (same logic as original)
    # ========================================================================

    prior_agg = prior_df.groupBy("mrn", "time_bucket_label").agg(
        F.max("diarrhea_present").alias("diarrhea_status"),
        F.max("severity_level").alias("severity"),
    ).withColumn(
        "diarrhea_status_formatted",
        F.when(
            (F.col("diarrhea_status") == "Yes")
            & F.col("severity").isNotNull()
            & (F.col("severity") != "N/A"),
            F.concat(F.lit("Yes (Lvl "), F.col("severity"), F.lit(")"))
        ).when(
            F.col("diarrhea_status") == "Yes", F.lit("Yes")
        ).otherwise(F.lit("No"))
    )

    # ========================================================================
    # STEP 8: Pivot the two new buckets into columns
    # ========================================================================

    prior_pivot = prior_agg.groupBy("mrn").pivot(
        "time_bucket_label", ["-12_months", "-6_months"]
    ).agg(
        F.first("diarrhea_status_formatted")
    )

    # ========================================================================
    # STEP 9: Join new columns onto the existing data rows
    # ========================================================================

    data_rows = data_rows.join(prior_pivot, on="mrn", how="left")

    # ========================================================================
    # STEP 10: Reorder columns so -12_months and -6_months come before 0_months
    # ========================================================================

    # Identify all time columns (existing + new)
    all_time_cols = sorted(
        [c for c in data_rows.columns if c.endswith("_months")],
        key=lambda x: int(x.replace("_months", ""))
    )

    meta_cols = ["mrn", "enrollment_start_date", "enrollment_end_date"]
    final_col_order = meta_cols + all_time_cols

    data_rows = data_rows.select(*final_col_order)

    # ========================================================================
    # STEP 11: Rebuild SUMMARY row with positive/total counts for ALL columns
    # ========================================================================

    summary_data = {
        "mrn":                   "SUMMARY",
        "enrollment_start_date": None,
        "enrollment_end_date":   None,
    }

    for col_name in all_time_cols:
        total_col    = data_rows.filter(F.col(col_name).isNotNull()).count()
        positive_col = data_rows.filter(F.col(col_name).startswith("Yes")).count()
        summary_data[col_name] = f"{positive_col}/{total_col}"

    schema_fields = [
        StructField("mrn",                   StringType(), True),
        StructField("enrollment_start_date", StringType(), True),
        StructField("enrollment_end_date",   StringType(), True),
    ] + [StructField(c, StringType(), True) for c in all_time_cols]

    summary_df = data_rows.sparkSession.createDataFrame(
        [summary_data], StructType(schema_fields)
    )

    # ========================================================================
    # STEP 12: Final output — cast date cols to string, union with summary
    # ========================================================================

    df_final = data_rows.select(
        "mrn",
        F.col("enrollment_start_date").cast("string"),
        F.col("enrollment_end_date").cast("string"),
        *all_time_cols
    ).union(summary_df)

    df_final = df_final.orderBy(
        F.when(F.col("mrn") == "SUMMARY", 1).otherwise(0),
        "mrn"
    )

    return df_final