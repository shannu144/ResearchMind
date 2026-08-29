import React, { useState } from 'react';
import { 
  Sparkles, 
  Search, 
  GitCompare, 
  BookOpen, 
  Share2, 
  ShieldAlert, 
  CheckCircle2, 
  Layers,
  ArrowRight,
  ExternalLink,
  FlaskConical,
  FileCode,
  Copy,
  Download,
  Check,
  Atom,
  Binary,
  Microscope,
  Compass,
  Lightbulb,
  Cpu
} from 'lucide-react';
import { GenAIAPI } from '../services/api';

const TABS = [
  { 
    id: 'gaps', 
    label: 'Research Gaps', 
    icon: Compass, 
    badge: 'Discovery', 
    color: 'from-amber-500/20 to-rose-500/20 text-amber-300 border-amber-500/30' 
  },
  { 
    id: 'comparator', 
    label: 'Paper Comparator', 
    icon: GitCompare, 
    badge: 'Cross-Doc', 
    color: 'from-blue-500/20 to-indigo-500/20 text-blue-300 border-blue-500/30' 
  },
  { 
    id: 'review', 
    label: 'Literature Review', 
    icon: BookOpen, 
    badge: 'Synthesis', 
    color: 'from-purple-500/20 to-pink-500/20 text-purple-300 border-purple-500/30' 
  },
  { 
    id: 'hypotheses', 
    label: 'Hypothesis Lab', 
    icon: Microscope, 
    badge: 'Scientific', 
    color: 'from-emerald-500/20 to-teal-500/20 text-emerald-300 border-emerald-500/30' 
  },
  { 
    id: 'citations', 
    label: 'Citation Network', 
    icon: Share2, 
    badge: 'Graph', 
    color: 'from-cyan-500/20 to-blue-500/20 text-cyan-300 border-cyan-500/30' 
  },
  { 
    id: 'bibtex', 
    label: 'BibTeX Exporter', 
    icon: FileCode, 
    badge: 'LaTeX', 
    color: 'from-indigo-500/20 to-purple-500/20 text-indigo-300 border-indigo-500/30' 
  },
];

export default function GenAISuite({ documents }) {
  const [activeTab, setActiveTab] = useState('gaps');

  // 1. Gaps State
  const [focusArea, setFocusArea] = useState('Transformer architectures, memory efficiency, and long context reasoning');
  const [gapsLoading, setGapsLoading] = useState(false);
  const [gapsResult, setGapsResult] = useState(null);

  // 2. Comparator State
  const [compDocA, setCompDocA] = useState('');
  const [compDocB, setCompDocB] = useState('');
  const [comparatorLoading, setComparatorLoading] = useState(false);
  const [comparatorResult, setComparatorResult] = useState(null);

  // 3. Lit Review State
  const [reviewTopic, setReviewTopic] = useState('Recent advances and challenges in deep learning and NLP architectures');
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewResult, setReviewResult] = useState(null);

  // 4. Citation Network State
  const [networkLoading, setNetworkLoading] = useState(false);
  const [networkResult, setNetworkResult] = useState(null);

  // 5. Hypothesis State
  const [domain, setDomain] = useState('Linear attention mechanisms and memory-efficient transformers');
  const [hypoLoading, setHypoLoading] = useState(false);
  const [hypoResult, setHypoResult] = useState(null);

  // 6. BibTeX State
  const [bibtexLoading, setBibtexLoading] = useState(false);
  const [bibtexResult, setBibtexResult] = useState(null);
  const [copied, setCopied] = useState(false);

  // Handlers
  const handleFindGaps = async () => {
    setGapsLoading(true);
    try {
      const res = await GenAIAPI.findGaps({
        focus_area: focusArea,
        top_k_chunks: 8,
      });
      setGapsResult(res.data);
    } catch (err) {
      console.error('Find gaps failed', err);
    } finally {
      setGapsLoading(false);
    }
  };

  const handleCompareDocs = async () => {
    if (!compDocA || !compDocB || compDocA === compDocB) return;
    setComparatorLoading(true);
    try {
      const res = await GenAIAPI.compareDocuments({
        document_ids: [parseInt(compDocA), parseInt(compDocB)],
        comparison_aspects: [
          'Core Methodology & Architecture',
          'Key Findings & Benchmarks',
          'Identified Limitations & Future Directions'
        ]
      });
      setComparatorResult(res.data);
    } catch (err) {
      console.error('Compare docs failed', err);
    } finally {
      setComparatorLoading(false);
    }
  };

  const handleGenerateReview = async () => {
    setReviewLoading(true);
    try {
      const res = await GenAIAPI.literatureReview({
        research_topic: reviewTopic,
        max_sections: 4,
      });
      setReviewResult(res.data);
    } catch (err) {
      console.error('Lit review failed', err);
    } finally {
      setReviewLoading(false);
    }
  };

  const handleBuildCitationNetwork = async () => {
    setNetworkLoading(true);
    try {
      const res = await GenAIAPI.citationNetwork();
      setNetworkResult(res.data);
    } catch (err) {
      console.error('Citation graph failed', err);
    } finally {
      setNetworkLoading(false);
    }
  };

  const handleGenerateHypotheses = async () => {
    setHypoLoading(true);
    try {
      const res = await GenAIAPI.generateHypotheses({
        research_domain: domain,
        top_k: 6,
      });
      setHypoResult(res.data);
    } catch (err) {
      console.error('Hypothesis generation failed', err);
    } finally {
      setHypoLoading(false);
    }
  };

  const handleExportBibtex = async () => {
    setBibtexLoading(true);
    try {
      const res = await GenAIAPI.exportBibtex();
      setBibtexResult(res.data);
    } catch (err) {
      console.error('Bibtex export failed', err);
    } finally {
      setBibtexLoading(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header with Title & Live Stats */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <span className="p-2 rounded-xl bg-gradient-to-tr from-purple-600/30 to-indigo-500/20 border border-purple-500/40 text-purple-300">
              <Sparkles className="w-5 h-5 text-purple-400" />
            </span>
            <span>GenAI Scientific Intelligence Suite</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Autonomous multi-paper synthesis, algorithmic gap detection, scientific hypothesis formulation, and citation exports.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="badge badge-purple flex items-center gap-1.5 py-1 px-2.5">
            <Atom className="w-3.5 h-3.5" />
            6 AI Workbenches
          </span>
        </div>
      </div>

      {/* Modern Horizontal Navigation Pill Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 p-1.5 rounded-2xl bg-slate-900/80 border border-white/[0.08] backdrop-blur-xl">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex flex-col items-center justify-center p-3 rounded-xl transition-all text-xs font-semibold gap-1.5 ${
                isActive
                  ? `bg-gradient-to-b ${tab.color} border shadow-lg text-white`
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-1.5">
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
              </div>
              <span className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                isActive ? 'bg-black/30 text-white/90' : 'text-slate-500'
              }`}>
                {tab.badge}
              </span>
            </button>
          );
        })}
      </div>

      {/* Tab 1: Research Gap Finder */}
      {activeTab === 'gaps' && (
        <div className="glass-panel p-6 space-y-6">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Compass className="w-5 h-5 text-amber-400" />
                Autonomous Research Gap & Opportunity Finder
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Analyzes multi-document limitation paragraphs, future work sections, and methodology bottlenecks to pinpoint under-explored problems.
              </p>
            </div>
            <span className="badge badge-amber font-mono">Dual-Angle Retrieval</span>
          </div>

          <div className="flex gap-3">
            <input
              type="text"
              value={focusArea}
              onChange={(e) => setFocusArea(e.target.value)}
              placeholder="e.g. Memory efficient self-attention, context window scaling..."
              className="input-field flex-1"
            />
            <button
              onClick={handleFindGaps}
              disabled={gapsLoading}
              className="btn-primary shrink-0"
            >
              <Sparkles className={`w-4 h-4 ${gapsLoading ? 'animate-spin' : ''}`} />
              <span>{gapsLoading ? 'Synthesizing...' : 'Discover Gaps'}</span>
            </button>
          </div>

          {gapsResult && (
            <div className="space-y-4 pt-2">
              <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800 pb-2">
                <span>Identified <strong>{gapsResult.identified_gaps.length}</strong> strategic research gaps</span>
                <span>Sources consulted: <strong>{gapsResult.sources_consulted}</strong> chunks</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {gapsResult.identified_gaps.map((gap, i) => (
                  <div key={i} className="glass-card p-5 space-y-3 border-amber-500/20 hover:border-amber-500/40 transition">
                    <div className="flex items-start gap-2.5">
                      <span className="p-1 rounded-md bg-amber-500/20 text-amber-400 text-xs font-mono font-bold">
                        #{i + 1}
                      </span>
                      <h4 className="text-sm font-bold text-amber-200 leading-snug">
                        {gap.gap_title}
                      </h4>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      {gap.description}
                    </p>
                    <div className="pt-2 border-t border-slate-800/80">
                      <div className="text-[11px] text-slate-500 font-medium mb-1">Evidence Passages:</div>
                      <div className="flex flex-wrap gap-1.5">
                        {gap.evidence_sources.map((src, j) => (
                          <span key={j} className="text-[10px] px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800">
                            {src}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Paper Comparator */}
      {activeTab === 'comparator' && (
        <div className="glass-panel p-6 space-y-6">
          <div>
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <GitCompare className="w-5 h-5 text-blue-400" />
              Multi-Document Comparative Synthesis
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Side-by-side technical dissection contrasting architectures, experimental results, and empirical baselines.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-slate-400 font-medium block mb-1.5">Select Document A:</label>
              <select
                value={compDocA}
                onChange={(e) => setCompDocA(e.target.value)}
                className="input-field"
              >
                <option value="">Choose Document A...</option>
                {documents.map((d) => (
                  <option key={d.id} value={d.id}>{d.filename}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-400 font-medium block mb-1.5">Select Document B:</label>
              <select
                value={compDocB}
                onChange={(e) => setCompDocB(e.target.value)}
                className="input-field"
              >
                <option value="">Choose Document B...</option>
                {documents.map((d) => (
                  <option key={d.id} value={d.id}>{d.filename}</option>
                ))}
              </select>
            </div>
          </div>

          <button
            onClick={handleCompareDocs}
            disabled={comparatorLoading || !compDocA || !compDocB || compDocA === compDocB}
            className="btn-primary"
          >
            <GitCompare className={`w-4 h-4 ${comparatorLoading ? 'animate-spin' : ''}`} />
            <span>{comparatorLoading ? 'Synthesizing Comparison...' : 'Compare Selected Documents'}</span>
          </button>

          {comparatorResult && (
            <div className="space-y-6 pt-2">
              {/* Document Overview Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {comparatorResult.document_summaries.map((doc, i) => (
                  <div key={i} className="glass-card p-4 space-y-2 border-indigo-500/30">
                    <div className="text-xs font-bold text-indigo-300 font-mono">
                      DOCUMENT {i === 0 ? 'A' : 'B'}: {doc.filename}
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      {doc.summary}
                    </p>
                  </div>
                ))}
              </div>

              {/* Dimensions */}
              <div className="space-y-3">
                <h4 className="text-sm font-bold text-slate-200">Comparison by Analytical Dimensions</h4>
                {comparatorResult.dimension_comparisons.map((dim, i) => (
                  <div key={i} className="glass-card p-4 space-y-1.5">
                    <div className="text-xs font-bold text-cyan-300 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                      {dim.dimension}
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      {dim.comparison}
                    </p>
                  </div>
                ))}
              </div>

              {/* Overall Synthesis */}
              <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-500/30 space-y-2">
                <div className="text-xs font-bold text-indigo-200 flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4 text-indigo-400" />
                  Meta-Synthesis & Key Takeaways
                </div>
                <p className="text-xs text-slate-200 leading-relaxed">
                  {comparatorResult.overall_synthesis}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Literature Review Generator */}
      {activeTab === 'review' && (
        <div className="glass-panel p-6 space-y-6">
          <div>
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-purple-400" />
              Automated Comprehensive Literature Review
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Synthesizes an academic survey paper with structured sections, citations, and critical synthesis.
            </p>
          </div>

          <div className="flex gap-3">
            <input
              type="text"
              value={reviewTopic}
              onChange={(e) => setReviewTopic(e.target.value)}
              placeholder="e.g. Advances in deep learning architectures and NLP..."
              className="input-field flex-1"
            />
            <button
              onClick={handleGenerateReview}
              disabled={reviewLoading}
              className="btn-primary shrink-0"
            >
              <Sparkles className={`w-4 h-4 ${reviewLoading ? 'animate-spin' : ''}`} />
              <span>{reviewLoading ? 'Writing Survey...' : 'Generate Review'}</span>
            </button>
          </div>

          {reviewResult && (
            <div className="space-y-6 pt-2">
              <div className="p-5 rounded-xl bg-purple-950/20 border border-purple-500/30 space-y-2">
                <div className="text-xs font-bold text-purple-300 uppercase tracking-wider font-mono">
                  Abstract
                </div>
                <p className="text-xs text-slate-200 leading-relaxed">
                  {reviewResult.abstract}
                </p>
              </div>

              <div className="space-y-4">
                {reviewResult.sections.map((sec, i) => (
                  <div key={i} className="glass-card p-5 space-y-2.5">
                    <h4 className="text-sm font-bold text-indigo-200 flex items-center gap-2">
                      <span className="text-xs px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono">
                        §{i + 1}
                      </span>
                      {sec.section_title}
                    </h4>
                    <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-line">
                      {sec.content}
                    </p>
                    {sec.citations && sec.citations.length > 0 && (
                      <div className="pt-2 flex flex-wrap gap-1.5">
                        {sec.citations.map((c, j) => (
                          <span key={j} className="text-[10px] px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800 font-mono">
                            {c}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
                <div className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono">
                  Conclusion & Future Trajectories
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {reviewResult.conclusion}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 4: Scientific Hypothesis Generator */}
      {activeTab === 'hypotheses' && (
        <div className="glass-panel p-6 space-y-6">
          <div>
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Microscope className="w-5 h-5 text-emerald-400" />
              Automated Scientific Hypothesis & Experiment Designer
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Generates rigorous, testable hypotheses with formal variables, baseline models, benchmark datasets, and evaluation metrics.
            </p>
          </div>

          <div className="flex gap-3">
            <input
              type="text"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              placeholder="e.g. Memory efficient self-attention in low resource NLP..."
              className="input-field flex-1"
            />
            <button
              onClick={handleGenerateHypotheses}
              disabled={hypoLoading}
              className="btn-primary shrink-0"
            >
              <FlaskConical className={`w-4 h-4 ${hypoLoading ? 'animate-spin' : ''}`} />
              <span>{hypoLoading ? 'Formulating...' : 'Design Hypotheses'}</span>
            </button>
          </div>

          {hypoResult && (
            <div className="space-y-5 pt-2">
              <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800 pb-2">
                <span>Formulated <strong>{hypoResult.hypotheses.length}</strong> scientific hypotheses</span>
                <span>Analyzed <strong>{hypoResult.sources_analyzed}</strong> source passages</span>
              </div>

              <div className="space-y-4">
                {hypoResult.hypotheses.map((h, i) => (
                  <div key={i} className="glass-card p-5 space-y-4 border-emerald-500/20 hover:border-emerald-500/40 transition">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <span className="text-[10px] px-2 py-0.5 rounded font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                          HYPOTHESIS #{h.hypothesis_id}
                        </span>
                        <h4 className="text-base font-bold text-white mt-1.5">
                          {h.title}
                        </h4>
                      </div>
                    </div>

                    <div className="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1.5">
                      <div className="text-xs font-bold text-indigo-300 uppercase tracking-wider font-mono">
                        Formal Hypothesis Statement:
                      </div>
                      <p className="text-xs text-slate-200 font-medium leading-relaxed italic">
                        "{h.formal_hypothesis}"
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                      <div className="space-y-1">
                        <span className="text-slate-400 font-medium">Theoretical Rationale:</span>
                        <p className="text-slate-300 leading-relaxed">{h.rationale}</p>
                      </div>
                      <div className="space-y-1">
                        <span className="text-slate-400 font-medium">Expected Empirical Outcome:</span>
                        <p className="text-emerald-300 leading-relaxed">{h.expected_outcome}</p>
                      </div>
                    </div>

                    {/* Experiment Plan Matrix */}
                    <div className="p-3.5 rounded-xl bg-slate-950/70 border border-white/[0.05] space-y-3">
                      <div className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                        <FlaskConical className="w-3.5 h-3.5 text-emerald-400" />
                        Rigorous Experimental Validation Protocol
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
                        <div>
                          <div className="text-slate-500 font-mono">Independent Vars:</div>
                          <div className="text-slate-200 font-medium mt-0.5">
                            {h.experiment_plan.independent_variables.join(', ')}
                          </div>
                        </div>
                        <div>
                          <div className="text-slate-500 font-mono">Dependent Vars:</div>
                          <div className="text-slate-200 font-medium mt-0.5">
                            {h.experiment_plan.dependent_variables.join(', ')}
                          </div>
                        </div>
                        <div>
                          <div className="text-slate-500 font-mono">Baseline Models:</div>
                          <div className="text-slate-200 font-medium mt-0.5">
                            {h.experiment_plan.baseline_models.join(', ')}
                          </div>
                        </div>
                        <div>
                          <div className="text-slate-500 font-mono">Target Metrics:</div>
                          <div className="text-slate-200 font-medium mt-0.5">
                            {h.experiment_plan.evaluation_metrics.join(', ')}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 5: Citation Network Builder */}
      {activeTab === 'citations' && (
        <div className="glass-panel p-6 space-y-6">
          <div>
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Share2 className="w-5 h-5 text-cyan-400" />
              Semantic Citation & Concept Graph
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Constructs semantic similarity edges between documents in the corpus based on shared keyphrases and vector alignment.
            </p>
          </div>

          <button
            onClick={handleBuildCitationNetwork}
            disabled={networkLoading}
            className="btn-primary"
          >
            <Share2 className={`w-4 h-4 ${networkLoading ? 'animate-spin' : ''}`} />
            <span>{networkLoading ? 'Computing Network...' : 'Build Corpus Graph'}</span>
          </button>

          {networkResult && (
            <div className="space-y-6 pt-2">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="glass-card p-4 text-center">
                  <div className="text-xs text-slate-400">Total Nodes (Papers)</div>
                  <div className="text-xl font-bold text-cyan-400 mt-1">{networkResult.nodes.length}</div>
                </div>
                <div className="glass-card p-4 text-center">
                  <div className="text-xs text-slate-400">Semantic Edges</div>
                  <div className="text-xl font-bold text-indigo-400 mt-1">{networkResult.edges.length}</div>
                </div>
                <div className="glass-card p-4 text-center">
                  <div className="text-xs text-slate-400">Central Hub Document</div>
                  <div className="text-sm font-semibold text-white mt-1 truncate">
                    {networkResult.central_document || 'None'}
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="text-sm font-bold text-slate-200">Semantic Relationships</h4>
                {networkResult.edges.length === 0 ? (
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-400 text-center">
                    No strong semantic edges detected above threshold (upload more related documents to form graph).
                  </div>
                ) : (
                  <div className="space-y-2">
                    {networkResult.edges.map((edge, i) => (
                      <div key={i} className="glass-card p-4 flex items-center justify-between text-xs">
                        <div className="flex items-center gap-3">
                          <span className="font-semibold text-slate-200">{edge.source_doc}</span>
                          <ArrowRight className="w-3.5 h-3.5 text-cyan-400" />
                          <span className="font-semibold text-slate-200">{edge.target_doc}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="text-[10px] text-slate-400">
                            Concepts: <strong>{edge.shared_concepts.join(', ')}</strong>
                          </div>
                          <span className="badge badge-cyan">{(edge.similarity_score * 100).toFixed(1)}% Sim</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 6: BibTeX Exporter */}
      {activeTab === 'bibtex' && (
        <div className="glass-panel p-6 space-y-6">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <FileCode className="w-5 h-5 text-indigo-400" />
                Academic BibTeX & LaTeX Bibliography Exporter
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Generates fully compliant `.bib` citation entries from all ingested papers for direct use in Overleaf and LaTeX documents.
              </p>
            </div>
            <span className="badge badge-indigo font-mono">Overleaf Ready</span>
          </div>

          <button
            onClick={handleExportBibtex}
            disabled={bibtexLoading}
            className="btn-primary"
          >
            <FileCode className={`w-4 h-4 ${bibtexLoading ? 'animate-spin' : ''}`} />
            <span>{bibtexLoading ? 'Compiling Citations...' : 'Generate BibTeX Records'}</span>
          </button>

          {bibtexResult && (
            <div className="space-y-4 pt-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">
                  Compiled <strong>{bibtexResult.total_entries}</strong> citation records
                </span>
                <button
                  onClick={() => copyToClipboard(bibtexResult.bibtex_string)}
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied to Clipboard!' : 'Copy BibTeX'}</span>
                </button>
              </div>

              <div className="relative">
                <pre className="p-4 rounded-xl bg-slate-950 border border-white/[0.08] text-xs font-mono text-cyan-300 overflow-x-auto max-h-96 leading-relaxed">
                  {bibtexResult.bibtex_string}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
