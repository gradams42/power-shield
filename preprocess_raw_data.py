import pandas as pd
import glob
import os
import numpy as np

data_path = "binaryAllNaturalPlusNormalVsAttacks"
all_files = glob.glob(os.path.join(data_path, "data*.csv"))

li = []
for filename in all_files:
    # Use low_memory=False to prevent DtypeWarnings
    df = pd.read_csv(filename, low_memory=False)
    li.append(df)

full_df = pd.concat(li, axis=0, ignore_index=True)

# --- FIX START ---
# 1. Clean column names (remove leading/trailing spaces)
full_df.columns = full_df.columns.str.strip()

# 2. Extract 'marker' before numeric conversion
# In some versions of this dataset, it's 'marker', in others '@marker'
target_col = 'marker' if 'marker' in full_df.columns else '@marker'
y_raw = full_df[target_col].copy()

# 3. Convert only the FEATURE columns to numeric
X_raw = full_df.drop(columns=[target_col])
X_numeric = X_raw.apply(pd.to_numeric, errors='coerce')

# 4. Handle Infinity and NaNs in features
X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
X_numeric.fillna(X_numeric.median(), inplace=True)

# 5. Recombine
full_df = pd.concat([X_numeric, y_raw.rename('marker')], axis=1)
# --- FIX END ---

print(f"Merge Complete. Shape: {full_df.shape}")
print(f"Target distribution:\n{full_df['marker'].value_counts()}")

# Save as pickle for the next stage
full_df.to_pickle("processed_power_data.pkl")
