"""
Palantir Foundry PySpark Transform: Diarrhea Patient × Timepoint Matrix

Pipeline:
  1. Parse enrollment JSON → medication start date (time 0) per patient
  2. Parse extracted_llm JSON → baseline diarrhea present/severity per note
  3. Parse val_llm JSON → validator confirms true positives only
  4. Merge: val_llm positive overrides extractor; otherwise keep extractor result
  5. Bucket into 6-month intervals, pivot, summarize

Merge logic:
  - val_llm says "Yes"       → "Yes"  (verified positive)
  - val_llm is null/missing  → keep extracted_llm result (Yes/No/null)
  - val_llm says "No"        → "No"   (validator rejected the positive)
  - extracted_llm says "No"  → "No"   (extractor said negative, validator didn't touch it)
  - extracted_llm is null    → null   (no data)

FIX: Records without an explicit extractor-provided date are dropped.
     We no longer fall back to contact_date.
"""
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType, ArrayType
)
from transforms.api import transform_df, Input, Output

INTERVAL_MONTHS  = 6
MIN_TIME_BUCKET  = -6
MAX_TIME_BUCKET  = 36

INELIGIBLE_STATUSES = ["deceased_or_ineligible"]

_YES_VALUES = {"yes", "true", "present", "positive", "1"}
_NO_VALUES  = {"no", "false", "absent", "negative", "0", "none", "n/a"}


def _normalize_present(col_expr):
    """Normalize a 'present' field to exact 'Yes'/'No'/null."""
    lower = F.lower(F.trim(col_expr))
    return (
        F.when(lower.isin(list(_YES_VALUES)), F.lit("Yes"))
         .when(lower.isin(list(_NO_VALUES)), F.lit("No"))
         .otherwise(F.lit(None).cast("string"))
    )


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    input_data=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(input_data):

    # ========================================================================
    # STEP 1: Parse enrollment JSON
    # ========================================================================

    enrollment_schema = StructType([
        StructField("confirmed_start",             BooleanType(), True),
        StructField("confirmed_start_date",        StringType(),  True),
        StructField("confirmed_end",               BooleanType(), True),
        StructField("confirmed_end_date",          StringType(),  True),
        StructField("enrollment_status_this_note", StringType(),  True),
        StructField("confidence",                  StringType(),  True),
        StructField("evidence_text",               StringType(),  True),
    ])

    df = input_data.withColumn(
        "enrollment_cleaned",
        F.when(
            F.col("enrollment").contains("```json"),
            F.regexp_extract(F.col("enrollment"), r"```json\s*([\s\S]*?)\s*```", 1)
        ).otherwise(F.col("enrollment"))
    ).withColumn(
        "enrollment_cleaned",
        F.regexp_replace(
            F.regexp_replace(F.col("enrollment_cleaned"), r"\n", " "),
            r"\s+", " "
        )
    ).withColumn(
        "enrollment_parsed",
        F.from_json(F.col("enrollment_cleaned"), enrollment_schema)
    )

    df = df.withColumn("enroll_confirmed_start",      F.col("enrollment_parsed.confirmed_start")) \
           .withColumn("enroll_confirmed_start_date", F.col("enrollment_parsed.confirmed_start_date")) \
           .withColumn("enroll_confirmed_end",        F.col("enrollment_parsed.confirmed_end")) \
           .withColumn("enroll_confirmed_end_date",   F.col("enrollment_parsed.confirmed_end_date")) \
           .withColumn("enroll_status",               F.col("enrollment_parsed.enrollment_status_this_note")) \
           .withColumn("enroll_confidence",           F.col("enrollment_parsed.confidence"))

    df = df.withColumn(
        "enroll_start_date_parsed",
        F.to_date(F.col("enroll_confirmed_start_date"), "M/d/yy")
    ).withColumn(
        "enroll_end_date_parsed",
        F.to_date(F.col("enroll_confirmed_end_date"), "M/d/yy")
    )

    # ========================================================================
    # STEP 2a: Parse extracted_llm JSON → baseline diarrhea per note
    # This is the EXTRACTOR — has both positives and negatives
    # ========================================================================

    diarrhea_schema = ArrayType(StructType([
        StructField("phenotype",      StringType(), True),
        StructField("present",        StringType(), True),
        StructField("severity_level", StringType(), True),
        StructField("time",           StringType(), True),
        StructField("evidence_text",  StringType(), True),
    ]))

    df = df.withColumn(
        "extractor_cleaned",
        F.when(
            F.col("extracted_llm").contains("```json"),
            F.regexp_extract(F.col("extracted_llm"), r"```json\s*([\s\S]*?)\s*```", 1)
        ).otherwise(F.col("extracted_llm"))
    ).withColumn(
        "extractor_cleaned",
        F.regexp_replace(
            F.regexp_replace(F.col("extractor_cleaned"), r"\n", " "),
            r"\s+", " "
        )
    ).withColumn(
        "extractor_parsed",
        F.from_json(F.col("extractor_cleaned"), diarrhea_schema)
    )

    # Extract diarrhea entry from extractor
    df = df.withColumn(
        "extractor_diarrhea",
        F.expr("filter(extractor_parsed, x -> x.phenotype = 'Diarrhea')")[0]
    ).withColumn(
        "extractor_present",
        _normalize_present(F.col("extractor_diarrhea.present"))
    ).withColumn(
        "extractor_severity",
        F.col("extractor_diarrhea.severity_level")
    ).withColumn(
        "extractor_time",
        F.col("extractor_diarrhea.time")
    )

    # ========================================================================
    # STEP 2b: Parse val_llm JSON → validator (true positives only)
    # This is the VALIDATOR — only confirms positives; nulls/missing = not validated
    # ========================================================================

    df = df.withColumn(
        "validator_cleaned",
        F.when(
            F.col("val_llm").isNotNull() & F.col("val_llm").contains("```json"),
            F.regexp_extract(F.col("val_llm"), r"```json\s*([\s\S]*?)\s*```", 1)
        ).when(
            F.col("val_llm").isNotNull(),
            F.col("val_llm")
        ).otherwise(F.lit(None).cast("string"))
    ).withColumn(
        "validator_cleaned",
        F.when(
            F.col("validator_cleaned").isNotNull(),
            F.regexp_replace(
                F.regexp_replace(F.col("validator_cleaned"), r"\n", " "),
                r"\s+", " "
            )
        ).otherwise(F.lit(None).cast("string"))
    ).withColumn(
        "validator_parsed",
        F.when(
            F.col("validator_cleaned").isNotNull(),
            F.from_json(F.col("validator_cleaned"), diarrhea_schema)
        )
    )

    # Extract diarrhea entry from validator
    df = df.withColumn(
        "validator_diarrhea",
        F.when(
            F.col("validator_parsed").isNotNull(),
            F.expr("filter(validator_parsed, x -> x.phenotype = 'Diarrhea')")[0]
        )
    ).withColumn(
        "validator_present",
        F.when(
            F.col("validator_diarrhea.present").isNotNull(),
            _normalize_present(F.col("validator_diarrhea.present"))
        ).otherwise(F.lit(None).cast("string"))
    ).withColumn(
        "validator_severity",
        F.col("validator_diarrhea.severity_level")
    )

    # ========================================================================
    # STEP 2c: MERGE extractor + validator
    #
    # Logic:
    #   validator "Yes"                        → "Yes" (verified positive)
    #   validator "No"                         → "No"  (validator rejected)
    #   validator null + extractor "Yes"       → "No"  (unconfirmed positive = rejected)
    #   validator null + extractor "No"        → "No"
    #   validator null + extractor null        → null  (no data at all)
    #
    # In short: ONLY validator "Yes" → "Yes". Everything else is "No" or null.
    # ========================================================================

    df = df.withColumn(
        "diarrhea_present",
        F.when(
            F.col("validator_present") == "Yes",
            F.lit("Yes")
        ).when(
            F.col("validator_present") == "No",
            F.lit("No")
        ).when(
            # Validator null + extractor had something (Yes or No) → "No"
            F.col("extractor_present").isNotNull(),
            F.lit("No")
        ).otherwise(
            # Both null → truly no data
            F.lit(None).cast("string")
        )
    ).withColumn(
        "severity_level",
        F.when(
            F.col("validator_present") == "Yes",
            F.coalesce(F.col("validator_severity"), F.col("extractor_severity"))
        ).otherwise(
            F.col("extractor_severity")
        )
    )

    # ── DIAGNOSTIC: Show merge results ─────────────────────────────────────
    print("\n[DIAGNOSTIC] Extractor vs Validator vs Final merge:")
    df.groupBy("extractor_present", "validator_present", "diarrhea_present") \
        .count().orderBy("extractor_present", "validator_present").show(50, truncate=False)

    # ========================================================================
    # FIX: Only use the extractor-provided time. If the extractor didn't
    # supply a date, this record has no valid timepoint and must be dropped.
    # We no longer fall back to contact_date.
    # ========================================================================

    df = df.withColumn(
        "note_time",
        F.when(
            F.col("extractor_time").isNotNull() & (F.trim(F.col("extractor_time")) != ""),
            F.to_date(F.col("extractor_time"), "M/d/yy")
        ).otherwise(F.lit(None).cast("date"))
    )

    # Drop records where the extractor had no valid date
    df = df.filter(F.col("note_time").isNotNull())

    # ========================================================================
    # STEP 3: Aggregate enrollment signals to patient level
    # ========================================================================

    enrollment_agg = df.groupBy("mrn").agg(
        F.max(F.col("enroll_confirmed_start").cast("int")).cast("boolean")
            .alias("any_confirmed_start"),
        F.max(
            F.when(F.col("enroll_status").isin(INELIGIBLE_STATUSES), F.lit(1)).otherwise(F.lit(0))
        ).cast("boolean").alias("any_ineligible"),
        F.min(
            F.when(F.col("enroll_confirmed_start") == True, F.col("enroll_start_date_parsed"))
        ).alias("enrollment_start_date"),
        F.max(
            F.when(F.col("enroll_confirmed_end") == True, F.col("enroll_end_date_parsed"))
        ).alias("enrollment_end_date"),
    ).withColumn(
        "enrolled",
        F.col("any_confirmed_start") & ~F.col("any_ineligible")
    )

    total          = enrollment_agg.count()
    enrolled_count = enrollment_agg.filter(F.col("enrolled") == True).count()
    print(f"\n[Enrollment Gate] Total patients : {total}")
    print(f"[Enrollment Gate] Enrolled       : {enrolled_count}")
    print(f"[Enrollment Gate] Dropped        : {total - enrolled_count}")

    # ========================================================================
    # STEP 4: Filter to enrolled patients
    # ========================================================================

    enrolled_patients = enrollment_agg.filter(
        (F.col("enrolled") == True) & F.col("enrollment_start_date").isNotNull()
    ).select("mrn", "enrollment_start_date", "enrollment_end_date")

    df = df.join(enrolled_patients, on="mrn", how="inner")

    # ========================================================================
    # STEP 5: Filter to medication window
    # ========================================================================

    df = df.filter(
        F.col("note_time").isNotNull() &
        (F.col("note_time") >= F.col("enrollment_start_date")) &
        (
            F.col("enrollment_end_date").isNull() |
            (F.col("note_time") <= F.col("enrollment_end_date"))
        )
    )

    # ========================================================================
    # STEP 6: Bucket into 6-month intervals
    # ========================================================================

    df = df.withColumn(
        "months_from_start",
        F.months_between(F.col("note_time"), F.col("enrollment_start_date"))
    ).withColumn(
        "time_bucket",
        F.floor(F.col("months_from_start") / INTERVAL_MONTHS) * INTERVAL_MONTHS
    ).filter(
        (F.col("time_bucket") >= MIN_TIME_BUCKET) &
        (F.col("time_bucket") <= MAX_TIME_BUCKET)
    ).withColumn(
        "time_bucket_label",
        F.concat(F.col("time_bucket").cast("int").cast("string"), F.lit("_months"))
    )

    # ========================================================================
    # STEP 7: Aggregate by patient × time bucket
    # Uses explicit counting — not alphabetical max
    # ========================================================================

    df_agg = df.groupBy("mrn", "time_bucket", "time_bucket_label").agg(
        F.count("*").alias("note_count"),
        F.sum(
            F.when(F.col("diarrhea_present") == "Yes", F.lit(1)).otherwise(F.lit(0))
        ).alias("positive_count"),
        F.sum(
            F.when(F.col("diarrhea_present") == "No", F.lit(1)).otherwise(F.lit(0))
        ).alias("negative_count"),
        F.sum(
            F.when(F.col("diarrhea_present").isNull(), F.lit(1)).otherwise(F.lit(0))
        ).alias("null_count"),
        F.max(
            F.when(F.col("diarrhea_present") == "Yes", F.col("severity_level"))
        ).alias("severity"),
        F.first("enrollment_start_date").alias("enrollment_start_date"),
        F.first("enrollment_end_date").alias("enrollment_end_date"),
    )

    df_agg = df_agg.withColumn(
        "diarrhea_status",
        F.when(F.col("positive_count") > 0, F.lit("Yes"))
         .when(F.col("negative_count") > 0, F.lit("No"))
         .otherwise(F.lit(None).cast("string"))
    ).withColumn(
        "diarrhea_status_formatted",
        F.when(
            (F.col("diarrhea_status") == "Yes") &
            F.col("severity").isNotNull() &
            (F.col("severity") != "N/A"),
            F.concat(F.lit("Yes (Lvl "), F.col("severity"), F.lit(")"))
        ).when(
            F.col("diarrhea_status") == "Yes", F.lit("Yes")
        ).when(
            F.col("diarrhea_status") == "No", F.lit("No")
        ).otherwise(F.lit(None).cast("string"))
    )

    # ── DIAGNOSTIC ─────────────────────────────────────────────────────────
    print("\n[DIAGNOSTIC] Aggregated status per bucket:")
    df_agg.groupBy("time_bucket_label", "diarrhea_status").count() \
        .orderBy("time_bucket_label", "diarrhea_status").show(100, truncate=False)

    # ========================================================================
    # STEP 8: Pivot to patient × timepoint matrix
    # ========================================================================

    df_pivot = df_agg.groupBy("mrn").pivot("time_bucket_label").agg(
        F.first("diarrhea_status_formatted")
    )

    meta = df_agg.select(
        "mrn", "enrollment_start_date", "enrollment_end_date"
    ).distinct()

    df_pivot = df_pivot.join(meta, on="mrn", how="left")

    time_columns = sorted(
        [c for c in df_pivot.columns if c.endswith("_months")],
        key=lambda x: int(x.replace("_months", ""))
    )

    # ========================================================================
    # STEP 9: Summary row — positive / assessed
    # ========================================================================

    summary_data = {
        "mrn":                   "SUMMARY",
        "enrollment_start_date": None,
        "enrollment_end_date":   None,
    }

    print("\n[SUMMARY] Per-timepoint breakdown:")
    for col_name in time_columns:
        assessed = df_pivot.filter(F.col(col_name).isNotNull()).count()
        positive = df_pivot.filter(
            F.col(col_name).isNotNull() & F.col(col_name).startswith("Yes")
        ).count()
        negative = assessed - positive
        pct = (positive / assessed * 100) if assessed > 0 else 0
        summary_data[col_name] = f"{positive}/{assessed}"
        print(f"  {col_name:15s}: {positive:>3} positive / {assessed:>3} assessed "
              f"({pct:.1f}%)  |  {negative} negative")

    schema_fields = [
        StructField("mrn",                   StringType(), True),
        StructField("enrollment_start_date", StringType(), True),
        StructField("enrollment_end_date",   StringType(), True),
    ] + [StructField(c, StringType(), True) for c in time_columns]

    summary_df = df_pivot.sparkSession.createDataFrame(
        [summary_data], StructType(schema_fields)
    )

    # ========================================================================
    # STEP 10: Final output
    # ========================================================================

    df_final = df_pivot.select(
        "mrn",
        F.col("enrollment_start_date").cast("string"),
        F.col("enrollment_end_date").cast("string"),
        *time_columns
    ).union(summary_df)

    df_final = df_final.orderBy(
        F.when(F.col("mrn") == "SUMMARY", 1).otherwise(0),
        "mrn"
    )

    return df_final