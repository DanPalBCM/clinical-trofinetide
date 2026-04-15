"""
Palantir Foundry PySpark Transform: Extract Patient Notes Pre-Enrollment
Purpose: Collect all clinical notes from 1 year before enrollment_start_date
         up to enrollment_start_date for each patient.
"""
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructType, StructField
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.xxxxx"),
    enrollment=Input("ri.foundry.main.dataset.xxxxx"),
    notes=Input("ri.foundry.main.dataset.xxxxx"),
    mrn_patient_id_mapping=Input("ri.foundry.main.dataset.xxxxx")
)
def compute(enrollment, notes, mrn_patient_id_mapping):
    """
    Extract all clinical notes from 1 year before each patient's
    enrollment_start_date up to (and including) that date.
    """

    # -------------------------------------------------------------------------
    # Step 1: Get enrollment info — MRN and enrollment_start_date
    # -------------------------------------------------------------------------
    enrollment_df = enrollment.select(
        F.col("mrn").cast("string").alias("mrn"),
        F.to_date(F.col("enrollment_start_date"), "yyyy-MM-dd").alias("enrollment_start_date")
    ).distinct()

    enrollment_df = enrollment_df.withColumn(
        "lookback_date",
        F.date_sub(F.col("enrollment_start_date"), 365)
    )

    # -------------------------------------------------------------------------
    # Step 2: Get patient_id to MRN mapping
    # -------------------------------------------------------------------------
    mapping = mrn_patient_id_mapping.select(
        F.col("patient_id").cast("string").alias("patient_id"),
        F.col("mrn").cast("string").alias("mrn")
    ).distinct()

    # -------------------------------------------------------------------------
    # Step 3: Join enrollment with mapping to get patient_ids + date bounds
    # -------------------------------------------------------------------------
    patients_with_ids = enrollment_df.join(
        mapping,
        on="mrn",
        how="inner"
    ).select("patient_id", "mrn", "enrollment_start_date", "lookback_date").distinct()

    # -------------------------------------------------------------------------
    # Step 4: Compute a GLOBAL date floor to pre-filter notes before the join.
    # This is the key optimization — narrow the notes dataset early, similar
    # to Code 1's Sept 2022 hard-coded cutoff.
    # -------------------------------------------------------------------------
    global_lookback = patients_with_ids.agg(
        F.min("lookback_date").alias("global_min")
    ).collect()[0]["global_min"]

    global_enrollment_max = patients_with_ids.agg(
        F.max("enrollment_start_date").alias("global_max")
    ).collect()[0]["global_max"]

    # -------------------------------------------------------------------------
    # Step 5: Filter notes by note_type AND global date bounds BEFORE joining.
    # This drastically reduces the notes volume entering the join.
    # -------------------------------------------------------------------------
    allowed_note_types = ["Progress Notes", "Telephone Encounter", "H&P", "Letter"]

    notes_std = notes.withColumn(
        "patient_id", F.col("patient_id").cast("string")
    ).filter(
        (F.col("note_type_ip").isin(allowed_note_types)) &
        (F.col("contact_date") >= F.lit(global_lookback)) &
        (F.col("contact_date") <= F.lit(global_enrollment_max))
    )

    # -------------------------------------------------------------------------
    # Step 6: Filter notes to only relevant patient_ids BEFORE the full join.
    # Use a semi-join (left_semi) — keeps only matching notes rows without
    # adding any columns, avoiding row duplication from the join.
    # -------------------------------------------------------------------------
    notes_our_patients = notes_std.join(
        patients_with_ids.select("patient_id"),
        on="patient_id",
        how="left_semi"
    )

    # -------------------------------------------------------------------------
    # Step 7: Now do the real join to attach per-patient date bounds,
    # operating on the already-reduced notes set.
    # -------------------------------------------------------------------------
    notes_joined = notes_our_patients.join(
        patients_with_ids.select("patient_id", "enrollment_start_date", "lookback_date"),
        on="patient_id",
        how="inner"
    )

    # -------------------------------------------------------------------------
    # Step 8: Apply per-patient date window filter
    # -------------------------------------------------------------------------
    notes_filtered = notes_joined.filter(
        (F.col("contact_date") >= F.col("lookback_date")) &
        (F.col("contact_date") <= F.col("enrollment_start_date"))
    )

    # Drop the helper columns before dedup so they don't bloat the output
    notes_filtered = notes_filtered.drop("enrollment_start_date", "lookback_date")

    # -------------------------------------------------------------------------
    # Step 9: Drop duplicates
    # -------------------------------------------------------------------------
    final_notes = notes_filtered.dropDuplicates()

    # -------------------------------------------------------------------------
    # Step 10: Chunking (unchanged)
    # -------------------------------------------------------------------------
    MAX_CHARS = 600000

    all_cols = final_notes.columns

    chunk_schema = ArrayType(
        StructType([
            StructField("note_id", StringType(), True),
            StructField("note_text", StringType(), True),
        ])
    )

    @F.udf(chunk_schema)
    def chunk_note(note_id, note_text):
        if note_text is None:
            return [{"note_id": note_id, "note_text": note_text}]
        if len(note_text) <= MAX_CHARS:
            return [{"note_id": str(note_id), "note_text": note_text}]
        chunks = []
        total_chunks = (len(note_text) + MAX_CHARS - 1) // MAX_CHARS
        for i in range(total_chunks):
            chunk_text = note_text[i * MAX_CHARS:(i + 1) * MAX_CHARS]
            chunk_id = "{}_chunked_{}".format(str(note_id), i + 1)
            chunks.append({"note_id": chunk_id, "note_text": chunk_text})
        return chunks

    other_cols = [c for c in all_cols if c not in ("note_id", "note_text")]

    chunked = (
        final_notes
        .withColumn("_chunks", chunk_note(F.col("note_id"), F.col("note_text")))
        .drop("note_id", "note_text")
        .withColumn("_chunk", F.explode(F.col("_chunks")))
        .drop("_chunks")
        .withColumn("note_id", F.col("_chunk.note_id"))
        .withColumn("note_text", F.col("_chunk.note_text"))
        .drop("_chunk")
        .select(all_cols)
    )

    return chunked