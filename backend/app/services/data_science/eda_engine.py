import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from app.core.logging import logger
from app.schemas.data_science_schemas import (
    DatasetEDADetailResponse,
    CorrelationMatrixResponse,
    NumericalDistributionStats,
    CategoricalDistributionItem,
)


class EDAEngine:
    """
    Automated Data Science Engine for Tabular Datasets.
    Computes Pearson Correlation Matrices, Skewness/Kurtosis, and Categorical Distributions.
    """

    def analyze_dataset(
        self, file_path: str, document_id: int, filename: str
    ) -> DatasetEDADetailResponse:
        df = pd.read_csv(file_path)

        row_count = len(df)
        col_count = len(df.columns)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        missing_counts = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
        missing_percentages = {
            col: round((count / row_count) * 100, 2) if row_count > 0 else 0.0
            for col, count in missing_counts.items()
        }
        duplicate_rows = int(df.duplicated().sum())

        # 1. Pearson Correlation Matrix for Numerical Columns
        correlation_response: Optional[CorrelationMatrixResponse] = None
        if len(num_cols) >= 2:
            corr_df = df[num_cols].corr(method="pearson").fillna(0.0)
            # Replace NaNs or infinite values if any
            corr_df = corr_df.replace([np.inf, -np.inf], 0.0)
            matrix_values = [[round(float(val), 4) for val in row] for row in corr_df.values]
            correlation_response = CorrelationMatrixResponse(
                columns=num_cols,
                matrix=matrix_values,
            )

        # 2. Numerical Distributions with Skewness & Kurtosis
        num_distributions: List[NumericalDistributionStats] = []
        for col in num_cols:
            series = df[col].dropna()
            if len(series) > 0:
                skew_val = float(series.skew()) if len(series) > 2 else 0.0
                kurt_val = float(series.kurtosis()) if len(series) > 3 else 0.0
                num_distributions.append(
                    NumericalDistributionStats(
                        column=col,
                        mean=round(float(series.mean()), 4),
                        std=round(float(series.std()), 4) if len(series) > 1 else 0.0,
                        min=round(float(series.min()), 4),
                        quantile_25=round(float(series.quantile(0.25)), 4),
                        median=round(float(series.median()), 4),
                        quantile_75=round(float(series.quantile(0.75)), 4),
                        max=round(float(series.max()), 4),
                        skewness=round(skew_val, 4),
                        kurtosis=round(kurt_val, 4),
                    )
                )

        # 3. Categorical Distributions
        cat_distributions: Dict[str, List[CategoricalDistributionItem]] = {}
        for col in cat_cols:
            counts = df[col].value_counts().head(10)
            items = []
            for cat_name, cnt in counts.items():
                pct = round((cnt / row_count) * 100, 2) if row_count > 0 else 0.0
                items.append(
                    CategoricalDistributionItem(
                        category=str(cat_name),
                        count=int(cnt),
                        percentage=pct,
                    )
                )
            cat_distributions[col] = items

        return DatasetEDADetailResponse(
            document_id=document_id,
            filename=filename,
            row_count=row_count,
            column_count=col_count,
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
            missing_values=missing_counts,
            missing_percentage=missing_percentages,
            duplicate_rows=duplicate_rows,
            correlation_matrix=correlation_response,
            numerical_distributions=num_distributions,
            categorical_distributions=cat_distributions,
        )
