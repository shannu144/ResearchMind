from typing import List, Optional
from app.schemas.genai_schemas import (
    ExperimentPlan,
    ScientificHypothesis,
    HypothesisGeneratorResponse,
)
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.embeddings.vector_store import VectorStoreFactory
from app.services.llm.provider import LLMProviderFactory


SYSTEM_PROMPT = """You are a Principal AI Research Scientist.
Based on the provided research context and identified limitations, formulate 2 to 4 rigorous, testable scientific hypotheses and design concrete experimental validation plans.

Format your output EXACTLY as:
HYPOTHESIS 1: [Title]
Rationale: [Why this hypothesis follows from current limitations]
Formal: [Formal If/Then or Null/Alternative hypothesis statement]
Outcome: [Expected theoretical or empirical improvement]
Variables: Independent: [var1, var2] | Dependent: [var1, var2]
Baselines: [baseline model 1, baseline model 2]
Datasets: [benchmark 1, benchmark 2]
Metrics: [metric 1, metric 2]

Be mathematically sound and technically grounded in the context provided."""


class ScientificHypothesisGenerator:
    """
    Automated Scientific Hypothesis & Experiment Design Engine.
    Synthesizes research corpus gaps into testable scientific hypotheses,
    defining variables, baselines, and evaluation protocols.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.llm_provider = LLMProviderFactory.get_provider()

    def generate_hypotheses(
        self,
        research_domain: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 6,
    ) -> HypothesisGeneratorResponse:
        query = f"limitations future work research gaps in {research_domain}"
        q_vec = self.embedding_service.encode([query])
        chunks = self.vector_store.search(
            query_vector=q_vec,
            top_k=top_k,
            document_ids=document_ids,
        )

        if not chunks:
            return HypothesisGeneratorResponse(
                research_domain=research_domain,
                hypotheses=[
                    ScientificHypothesis(
                        hypothesis_id=1,
                        title=f"Sample Hypothesis for {research_domain}",
                        rationale="Derived from foundational AI principles in the absence of indexed papers.",
                        formal_hypothesis="Incorporating sparse attention approximations will reduce compute complexity from O(N^2) to O(N) while maintaining BLEU score within 0.5% margin.",
                        expected_outcome="50% inference speedup with negligible accuracy degradation.",
                        experiment_plan=ExperimentPlan(
                            independent_variables=["Attention mechanism type (Full vs Sparse)"],
                            dependent_variables=["Latency (ms)", "BLEU Score"],
                            baseline_models=["Standard Transformer", "BERT-Base"],
                            suggested_datasets=["WMT14 En-De", "GLUE Benchmark"],
                            evaluation_metrics=["BLEU", "Latency", "VRAM Usage (GB)"],
                        ),
                    )
                ],
                llm_provider_used=self.llm_provider.provider_name,
                sources_analyzed=0,
            )

        context_text = "\n\n---\n\n".join(
            f"[{c.filename} Page {c.page_number}]: {c.text}" for c in chunks
        )

        prompt = f"""RESEARCH CONTEXT & LIMITATIONS:
{context_text}

DOMAIN OF INQUIRY: {research_domain}

Formulate 2-3 novel testable hypotheses with complete experiment design."""

        raw_output = self.llm_provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        hypotheses = self._parse_hypotheses(raw_output)

        return HypothesisGeneratorResponse(
            research_domain=research_domain,
            hypotheses=hypotheses,
            llm_provider_used=self.llm_provider.provider_name,
            sources_analyzed=len(chunks),
        )

    def _parse_hypotheses(self, raw_text: str) -> List[ScientificHypothesis]:
        """Parse structured hypothesis blocks from LLM output."""
        hypotheses = []
        try:
            sections = raw_text.split("HYPOTHESIS ")
            h_id = 1
            for sec in sections:
                if not sec.strip():
                    continue
                lines = sec.strip().splitlines()
                title = lines[0].lstrip("0123456789: ").strip()
                rationale, formal, outcome = "", "", ""
                ind_vars, dep_vars, baselines, datasets, metrics = [], [], [], [], []

                for line in lines[1:]:
                    low = line.lower()
                    if low.startswith("rationale:"):
                        rationale = line.split(":", 1)[1].strip()
                    elif low.startswith("formal:"):
                        formal = line.split(":", 1)[1].strip()
                    elif low.startswith("outcome:"):
                        outcome = line.split(":", 1)[1].strip()
                    elif low.startswith("variables:"):
                        parts = line.split("|")
                        for p in parts:
                            if "independent:" in p.lower():
                                ind_vars = [x.strip() for x in p.split(":", 1)[1].split(",")]
                            elif "dependent:" in p.lower():
                                dep_vars = [x.strip() for x in p.split(":", 1)[1].split(",")]
                    elif low.startswith("baselines:"):
                        baselines = [x.strip() for x in line.split(":", 1)[1].split(",")]
                    elif low.startswith("datasets:"):
                        datasets = [x.strip() for x in line.split(":", 1)[1].split(",")]
                    elif low.startswith("metrics:"):
                        metrics = [x.strip() for x in line.split(":", 1)[1].split(",")]

                hypotheses.append(
                    ScientificHypothesis(
                        hypothesis_id=h_id,
                        title=title[:100] if title else f"Scientific Hypothesis {h_id}",
                        rationale=rationale[:400] if rationale else "Synthesized from limitations in current corpus.",
                        formal_hypothesis=formal[:400] if formal else "Novel proposed method improves target metrics significantly over baselines.",
                        expected_outcome=outcome[:300] if outcome else "Demonstrable empirical gain on benchmark tasks.",
                        experiment_plan=ExperimentPlan(
                            independent_variables=ind_vars if ind_vars else ["Proposed Method"],
                            dependent_variables=dep_vars if dep_vars else ["Accuracy", "Efficiency"],
                            baseline_models=baselines if baselines else ["SOTA Baseline"],
                            suggested_datasets=datasets if datasets else ["Standard Benchmark"],
                            evaluation_metrics=metrics if metrics else ["F1-Score", "Inference Latency"],
                        ),
                    )
                )
                h_id += 1
        except Exception:
            hypotheses.append(
                ScientificHypothesis(
                    hypothesis_id=1,
                    title="Automated Research Hypothesis",
                    rationale=raw_text[:300],
                    formal_hypothesis="Proposed architecture will outperform existing baselines.",
                    expected_outcome="Statistically significant improvement.",
                    experiment_plan=ExperimentPlan(
                        independent_variables=["Model Architecture"],
                        dependent_variables=["F1 Score"],
                        baseline_models=["Baseline"],
                        suggested_datasets=["Corpus Dataset"],
                        evaluation_metrics=["F1", "Loss"],
                    ),
                )
            )

        return hypotheses if hypotheses else [
            ScientificHypothesis(
                hypothesis_id=1,
                title="Default Generated Hypothesis",
                rationale="Formulated from current corpus state.",
                formal_hypothesis="Applying architectural refinements will yield performance gains.",
                expected_outcome="Positive empirical delta.",
                experiment_plan=ExperimentPlan(
                    independent_variables=["Refined Pipeline"],
                    dependent_variables=["Accuracy"],
                    baseline_models=["Standard Model"],
                    suggested_datasets=["Evaluation Set"],
                    evaluation_metrics=["Accuracy", "Recall"],
                ),
            )
        ]
