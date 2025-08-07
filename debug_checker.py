# --- FILE: debug_checker.py ---
# A small script to inspect the Parquet files for debugging.

import polars as pl
import os

# Make sure this path is IDENTICAL to the one in your gateway_server.py
COUNTS_PATH = "/app/HCMAIC2025/AICHALLENGE_OPENCUBEE_2/Repo/HoangNguyen/support_script/inference_results_rfdetr_json/object_counts.parquet"
OUTPUT_CSV_PATH = "debug_counts_output.csv"

def inspect_data():
    print("--- Starting Parquet Data Inspection ---")
    
    if not os.path.exists(COUNTS_PATH):
        print(f"!!! FATAL ERROR: Cannot find the parquet file at: {COUNTS_PATH}")
        return

    try:
        # Load the dataframe
        df = pl.read_parquet(COUNTS_PATH)
        print(f"Successfully loaded {COUNTS_PATH}. Shape: {df.shape}")
        print(f"Columns found: {df.columns}")

        # --- KEY DEBUG STEP ---
        # Let's check for a common condition, e.g., images with at least one person.
        # I will use your corrected column name "image_name" here.
        test_condition = pl.col("person") >= 1
        
        # Select the columns we care about for this test
        # IMPORTANT: Replace 'image_name' if it's different. Check the printed columns above.
        filtered_df = df.filter(test_condition).select(["image_name", "person", "car"])
        
        print(f"\nFound {filtered_df.height} rows where 'person >= 1'.")
        
        if filtered_df.height > 0:
            print("Showing first 5 matching rows:")
            print(filtered_df.head(5))
            
            # Save the first 100 results to a CSV file for you to inspect
            filtered_df.head(100).write_csv(OUTPUT_CSV_PATH)
            print(f"\nSUCCESS: Saved 100 sample rows to '{OUTPUT_CSV_PATH}'.")
            print("Please check this CSV file to confirm the 'image_name' format and object counts.")
        else:
            print("\nWARNING: No rows matched the test condition 'person >= 1'.")
            print("This might indicate an issue with the data in the 'person' column.")

    except Exception as e:
        print(f"\n!!! An error occurred during inspection: {e}")
        print("Please double-check the column names and file path.")

if __name__ == "__main__":
    inspect_data()