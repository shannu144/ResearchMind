import React, { useState } from 'react';
import { 
  MessageSquare, 
  Send, 
  FileText, 
  ShieldCheck, 
  Sparkles, 
  Sliders, 
  ExternalLink,
  Info,
  Layers,
  Database,
  CheckCircle,
  BarChart2,
  Zap,
  Gauge,
  SlidersHorizontal
} from 'lucide-react';
import { RAGAPI } from '../services/api';

export default function RAGStudio({ documents }) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(4);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.20);
  const [selectedDocId, setSelectedDocId] = useState('');
  const [useHybrid, setUseHybrid] = useState(true);
  const [denseWeight, setDenseWeight] = useState(0.5);
  const [evaluateTriad, setEvaluateTriad] = useState(true);
  const [loading, setLoading] = useState(false);
  const [ragResponse, setRagResponse] = useState(null);

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const docIds = selectedDocId ? [parseInt(selectedDocId)] : null;
      const res = await RAGAPI.query({
        question: query,
        top_k: parseInt(topK),
        similarity_threshold: parseFloat(similarityThreshold),
        document_ids: docIds,
        use_hybrid_search: useHybrid,
        dense_weight: parseFloat(denseWeight),
        sparse_weight: parseFloat((1.0 - denseWeight).toFixed(2)),
        evaluate_triad: evaluateTriad,
      });
      setRagResponse(res.data);
    } catch (err) {
      console.error('RAG query failed', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <span className="p-2 rounded-xl bg-gradient-to-tr from-indigo-600/30 to-cyan-500/20 border border-indigo-500/40 text-indigo-300">
              <MessageSquare className="w-5 h-5 text-indigo-400" />
            </span>
            <span>RAG Studio & Citation Explorer</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Hybrid Okapi BM25 + FAISS Dense retrieval with anti-hallucination thresholds and automated RAG Triad quality evaluation.
          </p>
        </div>
        <span className="badge badge-indigo text-xs py-1 px-3">
          Search Mode: {useHybrid ? 'Hybrid RRF' : 'Dense FAISS'}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* Left 2 Cols: Query Input + Grounded Answer Output */}
        <div className="lg:col-span-2 space-y-6">
          <div className="glass-panel p-6 space-y-4">
            <form onSubmit={handleQuery} className="space-y-4">
              <div>
                <label className="text-xs text-slate-300 font-semibold block mb-1.5 flex items-center justify-between">
                  <span>Enter Scientific Query:</span>
                  <span className="text-[11px] text-slate-500 font-normal">Grounded exclusively on indexed passages</span>
                </label>
                <div className="flex gap-2.5">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. What is the computational complexity of the self-attention mechanism?"
                    className="input-field flex-1 text-xs sm:text-sm"
                  />
                  <button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="btn-primary shrink-0"
                  >
                    <Send className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    <span>{loading ? 'Retrieving...' : 'Ask RAG'}</span>
                  </button>
                </div>
              </div>

              {/* Scope & Mode Filters Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 pt-2 border-t border-white/[0.06] text-xs">
                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Document Scope:</label>
                  <select
                    value={selectedDocId}
                    onChange={(e) => setSelectedDocId(e.target.value)}
                    className="input-field py-1.5 text-xs"
                  >
                    <option value="">All Uploaded Documents</option>
                    {documents.map((d) => (
                      <option key={d.id} value={d.id}>{d.filename}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Top-K Passages:</label>
                  <select
                    value={topK}
                    onChange={(e) => setTopK(e.target.value)}
                    className="input-field py-1.5 text-xs"
                  >
                    <option value="2">2 Chunks</option>
                    <option value="4">4 Chunks</option>
                    <option value="6">6 Chunks</option>
                    <option value="8">8 Chunks</option>
                  </select>
                </div>

                <div className="flex items-center pt-5">
                  <label className="flex items-center gap-2 cursor-pointer select-none text-slate-300 text-xs">
                    <input
                      type="checkbox"
                      checked={useHybrid}
                      onChange={(e) => setUseHybrid(e.target.checked)}
                      className="accent-indigo-500 rounded"
                    />
                    <span className="flex items-center gap-1">
                      <Zap className="w-3.5 h-3.5 text-amber-400" />
                      Hybrid BM25
                    </span>
                  </label>
                </div>

                <div className="flex items-center pt-5">
                  <label className="flex items-center gap-2 cursor-pointer select-none text-slate-300 text-xs">
                    <input
                      type="checkbox"
                      checked={evaluateTriad}
                      onChange={(e) => setEvaluateTriad(e.target.checked)}
                      className="accent-indigo-500 rounded"
                    />
                    <span className="flex items-center gap-1">
                      <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                      RAG Triad
                    </span>
                  </label>
                </div>
              </div>

              {/* Hybrid Weight Slider (Only when Hybrid is Active) */}
              {useHybrid && (
                <div className="p-3.5 rounded-xl bg-slate-900/60 border border-white/[0.05] space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400 flex items-center gap-1.5">
                      <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-400" />
                      RRF Weighting Balance
                    </span>
                    <span className="font-mono text-cyan-300 font-semibold">
                      Dense: {(denseWeight * 100).toFixed(0)}% | Sparse BM25: {((1 - denseWeight) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.1"
                    value={denseWeight}
                    onChange={(e) => setDenseWeight(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
              )}
            </form>
          </div>

          {/* RAG Answer Display */}
          {ragResponse && (
            <div className="glass-panel p-6 space-y-5 border-indigo-500/40">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-indigo-400" />
                  <h3 className="text-base font-semibold text-white">Grounded Answer</h3>
                </div>
                <div className="flex items-center gap-2">
                  <span className="badge badge-indigo text-xs">{ragResponse.search_mode}</span>
                  <span className="badge badge-cyan text-xs">LLM: {ragResponse.llm_provider_used}</span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-slate-200 text-sm leading-relaxed whitespace-pre-line">
                {ragResponse.answer}
              </div>

              {/* RAG Triad Quality Scorecard */}
              {ragResponse.triad_score && (
                <div className="p-4 rounded-xl bg-gradient-to-r from-indigo-950/40 via-purple-950/20 to-slate-900/60 border border-indigo-500/30 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-indigo-200 flex items-center gap-1.5">
                      <Gauge className="w-4 h-4 text-indigo-400" />
                      LLMOps RAG Triad Quantitative Evaluation
                    </span>
                    <span className={`badge ${
                      ragResponse.triad_score.overall_triad_score >= 0.7 
                        ? 'badge-emerald' 
                        : ragResponse.triad_score.overall_triad_score >= 0.4 
                        ? 'badge-amber' 
                        : 'badge-rose'
                    }`}>
                      Score: {(ragResponse.triad_score.overall_triad_score * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="p-2.5 rounded-lg bg-slate-900/80 border border-white/[0.04]">
                      <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Context Relevance</div>
                      <div className="text-sm font-bold text-cyan-300 mt-1 font-mono">
                        {(ragResponse.triad_score.context_relevance * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-900/80 border border-white/[0.04]">
                      <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Faithfulness</div>
                      <div className="text-sm font-bold text-emerald-300 mt-1 font-mono">
                        {(ragResponse.triad_score.faithfulness * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-900/80 border border-white/[0.04]">
                      <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Answer Relevance</div>
                      <div className="text-sm font-bold text-indigo-300 mt-1 font-mono">
                        {(ragResponse.triad_score.answer_relevance * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Col: Source Citations */}
        <div className="space-y-4">
          <div className="glass-panel p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <FileText className="w-4 h-4 text-cyan-400" />
                Retrieved Source Passages
              </h3>
              {ragResponse && (
                <span className="badge badge-cyan text-[10px] font-mono">
                  {ragResponse.sources.length} Chunks
                </span>
              )}
            </div>

            {!ragResponse ? (
              <div className="py-12 text-center text-slate-500 border border-dashed border-slate-800 rounded-xl text-xs space-y-2">
                <FileText className="w-8 h-8 mx-auto text-slate-700" />
                <p>Run a query to retrieve source passages with cosine similarity scores and page numbers.</p>
              </div>
            ) : ragResponse.sources.length === 0 ? (
              <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/30 text-amber-300 text-xs">
                No passages met the similarity threshold ({similarityThreshold}).
              </div>
            ) : (
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
                {ragResponse.sources.map((src, i) => (
                  <div key={i} className="glass-card p-4 space-y-2 border-slate-800 hover:border-slate-700 transition">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-white truncate max-w-[140px]" title={src.filename}>
                        {src.filename}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-slate-400">p.{src.page_number}</span>
                        <span className="badge badge-cyan text-[10px]">
                          {(src.similarity_score * 100).toFixed(0)}% Match
                        </span>
                      </div>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed italic bg-slate-950/50 p-2.5 rounded-lg border border-white/[0.04]">
                      "{src.text_snippet}"
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
