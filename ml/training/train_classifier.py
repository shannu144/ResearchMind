import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend")))

from app.services.ml.model_trainer import DocumentClassifierPipeline

if __name__ == "__main__":
    print("=" * 60)
    print("ResearchMind — Training & Benchmarking Traditional ML Models")
    print("Models: Logistic Regression vs Random Forest vs SVM")
    print("=" * 60)

    pipeline = DocumentClassifierPipeline(models_dir="ml/models")
    res = pipeline.train_and_evaluate(test_size=0.2, random_state=42)

    print("\nTraining & Evaluation Complete!")
    print(f"Best Performing Model: {res.comparison.best_model_name.upper()} (F1-Score: {res.comparison.best_f1_score:.4f})\n")

    for name, metric in res.comparison.metrics.items():
        print(f"--- Model: {name.upper()} ---")
        print(f"  Accuracy:  {metric.accuracy:.4f}")
        print(f"  Precision: {metric.precision:.4f}")
        print(f"  Recall:    {metric.recall:.4f}")
        print(f"  F1-Score:  {metric.f1_score:.4f}")
        print("  Confusion Matrix:")
        for row in metric.confusion_matrix:
            print("   ", row)
        print()

    print("Saved Model Artifacts:")
    for m_name, path in res.saved_model_paths.items():
        print(f"  - {m_name}: {path}")
    print("=" * 60)
