import os
import pandas as pd
from src.data.tidy_data import performance_df, extract_complexity_df, calculate_differences, calculate_score_differences

# Crear la carpeta "data/results" si no existe
output_dir = os.path.join('data', 'results')
os.makedirs(output_dir, exist_ok=True)

# Ejecutar las funciones y obtener los DataFrames
df_performance = performance_df()
df_complexity = extract_complexity_df()
df_differences = calculate_differences(df_complexity)
df_score_differences = calculate_score_differences(df_performance, df_complexity)

# Guardar los DataFrames en formato Parquet
df_performance.to_parquet(os.path.join(output_dir, 'performance_data.parquet'))
df_complexity.to_parquet(os.path.join(output_dir, 'complexity_data.parquet'))
df_differences.to_parquet(os.path.join(output_dir, 'differences_data.parquet'))
df_score_differences.to_parquet(os.path.join(output_dir, 'score_differences_data.parquet'))

print("Results saved successfully in 'data/results' as Parquet files.")
