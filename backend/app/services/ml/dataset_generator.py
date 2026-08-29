from typing import List, Tuple

# Research domains for multi-class classification
LABELS = [
    "Computer Science & AI",
    "Medical & Life Sciences",
    "Financial Analytics",
    "Physics & Quantum",
    "Climate & Environment",
]

RAW_DATASET: List[Tuple[str, str]] = [
    # Computer Science & AI
    ("Deep neural networks, Transformers, self-attention mechanisms, and large language models for NLP.", "Computer Science & AI"),
    ("Convolutional neural networks for computer vision, object detection, and image classification algorithms.", "Computer Science & AI"),
    ("Reinforcement learning with Markov decision processes, Q-learning, and policy gradient optimization.", "Computer Science & AI"),
    ("Graph neural networks, node classification, edge prediction, and relational database indexing.", "Computer Science & AI"),
    ("Distributed computing systems, microservice architectures, fault tolerance, and API latency optimization.", "Computer Science & AI"),
    ("Supervised learning algorithms including decision trees, random forests, and support vector machines.", "Computer Science & AI"),

    # Medical & Life Sciences
    ("Clinical trials investigating monoclonal antibodies for targeted cancer immunotherapy treatments.", "Medical & Life Sciences"),
    ("Genomic sequencing, CRISPR gene editing, DNA mutation identification, and cellular pathology.", "Medical & Life Sciences"),
    ("Cardiovascular disease risk factors, hypertension, arterial stiffness, and echocardiogram diagnostics.", "Medical & Life Sciences"),
    ("Pharmacokinetics of novel antiviral drugs, viral replication inhibition, and enzyme kinetics.", "Medical & Life Sciences"),
    ("Neurological disorders, synaptic plasticity, neurodegenerative markers, and MRI brain scan analysis.", "Medical & Life Sciences"),
    ("Epidemiological modeling of infectious disease transmission rates and vaccine efficacy trials.", "Medical & Life Sciences"),

    # Financial Analytics
    ("Algorithmic trading strategies, high-frequency market order execution, and arbitrage opportunities.", "Financial Analytics"),
    ("Portfolio optimization using Markowitz mean-variance framework and Sharpe ratio maximization.", "Financial Analytics"),
    ("Credit risk assessment models using logistic regression, default probability, and credit scoring.", "Financial Analytics"),
    ("Option pricing theory, Black-Scholes partial differential equations, and implied volatility surfaces.", "Financial Analytics"),
    ("Macroeconomic inflation forecasting using vector autoregression and interest rate yield curves.", "Financial Analytics"),
    ("Corporate valuation, discounted cash flow modeling, financial statement audits, and balance sheet analysis.", "Financial Analytics"),

    # Physics & Quantum
    ("Quantum entanglement, superconducting qubits, quantum logic gates, and quantum error correction.", "Physics & Quantum"),
    ("Particle physics, Higgs boson decay channels, Large Hadron Collider particle acceleration experiments.", "Physics & Quantum"),
    ("General relativity, gravitational wave detection via laser interferometry, and black hole event horizons.", "Physics & Quantum"),
    ("Condensed matter physics, high-temperature superconductivity, and topological insulator phase transitions.", "Physics & Quantum"),
    ("Thermodynamics of non-equilibrium systems, statistical mechanics, and entropy generation.", "Physics & Quantum"),
    ("Optics, laser spectroscopy, photon emission dynamics, and electromagnetic field interactions.", "Physics & Quantum"),

    # Climate & Environment
    ("Global warming impact on polar ice sheet melting, atmospheric greenhouse gas concentrations, and sea level rise.", "Climate & Environment"),
    ("Ocean circulation patterns, El Niño Southern Oscillation, and marine ecosystem biodiversity preservation.", "Climate & Environment"),
    ("Renewable energy integration, solar photovoltaic efficiency, wind turbine aerodynamics, and battery storage.", "Climate & Environment"),
    ("Deforestation monitoring using satellite remote sensing imagery and forest carbon sequestration analysis.", "Climate & Environment"),
    ("Soil degradation, agricultural drought resilience, water resource management, and crop yield forecasting.", "Climate & Environment"),
    ("Air quality assessment, particulate matter emissions, atmospheric dispersion models, and urban pollution.", "Climate & Environment"),
]


def get_training_corpus() -> Tuple[List[str], List[str]]:
    """
    Returns annotated dataset tuple (texts, labels).
    """
    texts = [item[0] for item in RAW_DATASET]
    labels = [item[1] for item in RAW_DATASET]
    return texts, labels
