import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from app.core.config import settings
from app.core.logging import logger
from app.schemas.ml_schemas import (
    ModelEvaluationMetric,
    ModelComparisonResult,
    MLTrainResponse,
    PredictResponse,
    PredictionProbability,
)
from app.services.ml.vectorizer import TFIDFVectorizerService
from app.services.ml.dataset_generator import get_training_corpus, LABELS


class DocumentClassifierPipeline:
    """
    Machine Learning Document Classification Pipeline.
    Trains, evaluates, compares, serializes, and serves predictions for Logistic Regression, Random Forest, and SVM models.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

        self.vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        self.model_paths = {
            "logistic_regression": os.path.join(self.models_dir, "logistic_regression.pkl"),
            "random_forest": os.path.join(self.models_dir, "random_forest.pkl"),
            "svm": os.path.join(self.models_dir, "svm.pkl"),
        }

    def train_and_evaluate(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> MLTrainResponse:
        texts, labels = get_training_corpus()

        # Train/Test Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        # Feature Extraction via TF-IDF
        vec_service = TFIDFVectorizerService(max_features=5000, ngram_range=(1, 2))
        X_train = vec_service.fit_transform(X_train_raw)
        X_test = vec_service.transform(X_test_raw)

        # Save Vectorizer
        joblib.dump(vec_service.vectorizer, self.vectorizer_path)

        # Initialize Models
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, C=1.0, random_state=random_state),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "svm": SVC(kernel="linear", probability=True, random_state=random_state),
        }

        evaluations: Dict[str, ModelEvaluationMetric] = {}
        saved_paths: Dict[str, str] = {}
        best_model_name = ""
        best_f1 = -1.0

        classes_sorted = sorted(list(set(labels)))

        for name, model in models.items():
            # Fit model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate Evaluation Metrics
            acc = float(accuracy_score(y_test, y_pred))
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )

            cm = confusion_matrix(y_test, y_pred, labels=classes_sorted).tolist()

            metric = ModelEvaluationMetric(
                model_name=name,
                accuracy=round(acc, 4),
                precision=round(float(prec), 4),
                recall=round(float(rec), 4),
                f1_score=round(float(f1), 4),
                confusion_matrix=cm,
                labels=classes_sorted,
            )
            evaluations[name] = metric

            # Save Model Artifact
            save_p = self.model_paths[name]
            joblib.dump(model, save_p)
            saved_paths[name] = save_p

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name

        comparison = ModelComparisonResult(
            best_model_name=best_model_name,
            best_f1_score=round(best_f1, 4),
            metrics=evaluations,
        )

        logger.info(f"ML Model Training Completed. Best Model: {best_model_name} (F1: {best_f1:.4f})")
        return MLTrainResponse(
            message="Models successfully trained and benchmarked.",
            comparison=comparison,
            saved_model_paths=saved_paths,
        )

    def predict(self, text: str, model_name: str = "logistic_regression") -> PredictResponse:
        model_key = model_name.lower().replace(" ", "_")
        if model_key not in self.model_paths:
            model_key = "logistic_regression"

        model_path = self.model_paths[model_key]
        if not os.path.exists(self.vectorizer_path) or not os.path.exists(model_path):
            # Auto-train if models not saved yet
            self.train_and_evaluate()

        vectorizer = joblib.load(self.vectorizer_path)
        model = joblib.load(model_path)

        X_vec = vectorizer.transform([text])
        pred_label = str(model.predict(X_vec)[0])

        all_probs: List[PredictionProbability] = []
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_vec)[0]
            classes = model.classes_
            for cls_name, prob in zip(classes, probs):
                all_probs.append(
                    PredictionProbability(
                        category=str(cls_name), confidence=round(float(prob), 4)
                    )
                )
            # Sort by confidence descending
            all_probs.sort(key=lambda x: x.confidence, reverse=True)
            top_confidence = all_probs[0].confidence
        else:
            top_confidence = 1.0
            all_probs.append(PredictionProbability(category=pred_label, confidence=1.0))

        return PredictResponse(
            predicted_category=pred_label,
            confidence=top_confidence,
            all_probabilities=all_probs,
            model_used=model_key,
        )
