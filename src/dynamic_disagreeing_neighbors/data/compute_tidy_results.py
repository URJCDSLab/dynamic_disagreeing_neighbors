import os
import pandas as pd
from dynamic_disagreeing_neighbors.data.tidy_data import *

# Crear la carpeta "data/results" si no existe
output_dir = os.path.join('data', 'results')
os.makedirs(output_dir, exist_ok=True)

# Ejecutar las funciones y obtener los DataFrames
df_performance = performance_df()
df_complexity = extract_complexity_df()
df_differences = calculate_differences(df_complexity)
df_score_differences = calculate_score_differences(df_performance, df_complexity)
df_performance_results = extract_best_model_performance()
df_performance_results_detailed = extract_all_models_performance()
df_score_differences_all = calculate_score_differences_all(df_performance_results_detailed, df_complexity)
df_description = df_description()

# Guardar los DataFrames en formato Parquet
df_performance.to_parquet(os.path.join(output_dir, 'performance_data_agnostic.parquet'))
df_complexity.to_parquet(os.path.join(output_dir, 'complexity_data.parquet'))
df_differences.to_parquet(os.path.join(output_dir, 'differences_data.parquet'))
df_score_differences.to_parquet(os.path.join(output_dir, 'score_differences_data.parquet'))
df_performance_results.to_parquet(os.path.join(output_dir, 'performance_data.parquet'))
df_performance_results_detailed.to_parquet(os.path.join(output_dir, 'performance_detailed_data.parquet'))
df_score_differences_all.to_parquet(os.path.join(output_dir, 'score_differences_all_data.parquet'))
df_description.to_parquet(os.path.join(output_dir, 'df_description_data.parquet'))


print("Results saved successfully in 'data/results' as Parquet files.")
