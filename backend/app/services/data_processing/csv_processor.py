import numpy as np
import pandas as pd
from typing import Dict, Any, List
from app.core.logging import logger
from app.schemas.document_schemas import CSVPreprocessingResult


class CSVProcessor:
    """
    Automated Data Science & EDA Processor for Tabular CSV Datasets.
    Computes missing values, duplicate detection, column type classification, summary stats, and IQR outliers.
    """

    def analyze_csv(self, file_path: str, document_id: int = 0) -> CSVPreprocessingResult:
        df = pd.read_csv(file_path)

        row_count = len(df)
        col_count = len(df.columns)
        columns = list(df.columns)

        # Numerical vs Categorical
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Missing values analysis
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = {col: round((count / row_count) * 100, 2) if row_count > 0 else 0.0 for col, count in missing_counts.items()}

        # Duplicates
        duplicate_rows = int(df.duplicated().sum())

        # Summary statistics
        summary_stats: Dict[str, Dict[str, float]] = {}
        outliers_iqr: Dict[str, int] = {}

        for col in num_cols:
            series = df[col].dropna()
            if len(series) > 0:
                summary_stats[col] = {
                    "mean": float(round(series.mean(), 4)),
                    "std": float(round(series.std(), 4)) if len(series) > 1 else 0.0,
                    "min": float(round(series.min(), 4)),
                    "25%": float(round(series.quantile(0.25), 4)),
                    "50%": float(round(series.median(), 4)),
                    "75%": float(round(series.quantile(0.75), 4)),
                    "max": float(round(series.max(), 4)),
                }

                # Outlier detection using Interquartile Range (IQR)
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outliers_iqr[col] = int(len(outliers))
            else:
                summary_stats[col] = {}
                outliers_iqr[col] = 0

        return CSVPreprocessingResult(
            document_id=document_id,
            row_count=row_count,
            column_count=col_count,
            columns=columns,
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
            missing_values={k: int(v) for k, v in missing_counts.items()},
            missing_percentage=missing_percentages,
            duplicate_rows=duplicate_rows,
            summary_statistics=summary_stats,
            outliers_iqr=outliers_iqr,
        )
