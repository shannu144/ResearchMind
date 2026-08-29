import React, { useState, useEffect } from 'react';
import { 
  BarChart2, 
  PieChart, 
  TrendingUp, 
  Hash, 
  BookOpen, 
  Layers, 
  RefreshCw,
  Sparkles
} from 'lucide-react';
import { DataScienceAPI } from '../services/api';

export default function EDADashboard({ documents }) {
  const [corpusSummary, setCorpusSummary] = useState(null);
  const [selectedDocId, setSelectedDocId] = useState('');
  const [docEDA, setDocEDA] = useState(null);
  const [ngramData, setNgramData] = useState(null);
  const [ngramN, setNgramN] = useState(2);
  const [loading, setLoading] = useState(false);

  const fetchCorpusSummary = async () => {
    try {
      const res = await DataScienceAPI.corpusSummary();
      setCorpusSummary(res.data);
    } catch (err) {
      console.error('Failed to load corpus summary', err);
    }
  };

  const fetchDocEDA = async (docId) => {
    if (!docId) return;
    setLoading(true);
    try {
      const [edaRes, ngramRes] = await Promise.all([
        DataScienceAPI.eda(docId),
        DataScienceAPI.ngramAnalysis({ document_ids: [parseInt(docId)], n: ngramN, top_k: 12 }),
      ]);
      setDocEDA(edaRes.data);
      setNgramData(ngramRes.data);
    } catch (err) {
      console.error('Failed to fetch doc EDA', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCorpusSummary();
  }, []);

  useEffect(() => {
    if (documents && documents.length > 0 && !selectedDocId) {
      setSelectedDocId(documents[0].id.toString());
      fetchDocEDA(documents[0].id);
    }
  }, [documents]);

  const handleDocChange = (e) => {
    const docId = e.target.value;
    setSelectedDocId(docId);
    fetchDocEDA(docId);
  };

  const handleNgramChange = (n) => {
    setNgramN(n);
    if (selectedDocId) {
      DataScienceAPI.ngramAnalysis({ document_ids: [parseInt(selectedDocId)], n, top_k: 12 })
        .then(res => setNgramData(res.data))
        .catch(console.error);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
          <BarChart2 className="w-6 h-6 text-indigo-400" />
          Corpus Analytics & Exploratory Data Analysis
        </h2>
        <p className="text-sm text-slate-400">
          Statistical profiling, vocabulary richness (Type-Token Ratio), and N-Gram frequency distributions.
        </p>
      </div>

      {/* Metric Cards */}
      {corpusSummary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="glass-panel p-4 flex items-center gap-3.5">
            <div className="p-3 rounded-xl bg-indigo-950/60 border border-indigo-800/40 text-indigo-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs text-slate-400">Total Documents</div>
              <div className="text-xl font-bold text-white">{corpusSummary.total_documents}</div>
            </div>
          </div>

          <div className="glass-panel p-4 flex items-center gap-3.5">
            <div className="p-3 rounded-xl bg-cyan-950/60 border border-cyan-800/40 text-cyan-400">
              <Hash className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs text-slate-400">Corpus Word Count</div>
              <div className="text-xl font-bold text-white">{corpusSummary.total_words?.toLocaleString()}</div>
            </div>
          </div>

          <div className="glass-panel p-4 flex items-center gap-3.5">
            <div className="p-3 rounded-xl bg-emerald-950/60 border border-emerald-800/40 text-emerald-400">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs text-slate-400">Avg Words / Doc</div>
              <div className="text-xl font-bold text-white">{Math.round(corpusSummary.avg_words_per_doc)}</div>
            </div>
          </div>

          <div className="glass-panel p-4 flex items-center gap-3.5">
            <div className="p-3 rounded-xl bg-purple-950/60 border border-purple-800/40 text-purple-400">
              <TrendingUp className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs text-slate-400">Unique Vocabulary</div>
              <div className="text-xl font-bold text-white">{corpusSummary.total_unique_words?.toLocaleString()}</div>
            </div>
          </div>
        </div>
      )}

      {/* Document Selector */}
      <div className="flex items-center gap-3 bg-slate-900/60 p-3 rounded-xl border border-slate-800">
        <span className="text-xs font-semibold text-slate-300">Select Document for Deep EDA:</span>
        <select 
          value={selectedDocId} 
          onChange={handleDocChange}
          className="bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-1.5 outline-none focus:border-indigo-500"
        >
          {documents.map((d) => (
            <option key={d.id} value={d.id}>{d.filename}</option>
          ))}
        </select>
      </div>

      {loading ? (
        <div className="py-12 text-center text-slate-400">
          <RefreshCw className="w-6 h-6 animate-spin mx-auto text-indigo-500 mb-2" />
          <p className="text-sm">Computing lexical statistics & n-grams...</p>
        </div>
      ) : docEDA ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Lexical Richness & Statistics */}
          <div className="glass-panel p-5 space-y-4">
            <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              Lexical Metrics & Richness
            </h3>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs text-slate-400">Type-Token Ratio (TTR)</div>
                <div className="text-lg font-bold text-indigo-300">{(docEDA.type_token_ratio * 100).toFixed(1)}%</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Higher = richer vocabulary</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs text-slate-400">Avg Sentence Length</div>
                <div className="text-lg font-bold text-cyan-300">{docEDA.avg_sentence_length_words.toFixed(1)} words</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Syntactic complexity index</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs text-slate-400">Avg Word Length</div>
                <div className="text-lg font-bold text-emerald-300">{docEDA.avg_word_length_chars.toFixed(1)} chars</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Lexical sophistication</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs text-slate-400">Stopword Proportion</div>
                <div className="text-lg font-bold text-amber-300">{(docEDA.stopword_ratio * 100).toFixed(1)}%</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Function vs content words</div>
              </div>
            </div>

            {/* Top Frequent Terms Bar Chart representation */}
            <div className="space-y-2 pt-2">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Top Content Words</div>
              <div className="space-y-1.5">
                {docEDA.top_words.slice(0, 7).map((tw, idx) => {
                  const maxCount = docEDA.top_words[0].count;
                  const pct = Math.round((tw.count / maxCount) * 100);
                  return (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-300">{tw.word}</span>
                        <span className="text-indigo-400 font-semibold">{tw.count}</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div 
                          className="bg-gradient-to-r from-indigo-500 to-cyan-400 h-1.5 rounded-full"
                          style={{ width: `${pct}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* N-Gram Frequency Analyzer */}
          <div className="glass-panel p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                N-Gram Frequency Analyzer
              </h3>

              <div className="flex rounded-lg bg-slate-900 border border-slate-800 p-0.5 text-xs">
                {[1, 2, 3].map((n) => (
                  <button
                    key={n}
                    onClick={() => handleNgramChange(n)}
                    className={`px-2.5 py-1 rounded-md transition font-medium ${
                      ngramN === n ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {n === 1 ? 'Unigram' : n === 2 ? 'Bigram' : 'Trigram'}
                  </button>
                ))}
              </div>
            </div>

            {ngramData && ngramData.ngrams ? (
              <div className="space-y-2">
                {ngramData.ngrams.length === 0 ? (
                  <div className="text-xs text-slate-500 py-6 text-center">No n-grams found for this selection.</div>
                ) : (
                  ngramData.ngrams.map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 rounded-lg bg-slate-900/50 border border-slate-800/80 text-xs">
                      <span className="font-mono text-slate-300 font-medium">{item.ngram}</span>
                      <span className="badge badge-cyan font-mono">{item.frequency}</span>
                    </div>
                  ))
                )}
              </div>
            ) : (
              <div className="text-xs text-slate-500 py-6 text-center">Loading n-grams...</div>
            )}
          </div>
        </div>
      ) : (
        <div className="py-12 text-center text-slate-500 text-xs">
          Upload and select a document to inspect Exploratory Data Analysis.
        </div>
      )}
    </div>
  );
}
