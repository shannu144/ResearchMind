import React, { useState } from 'react';
import { 
  Search, 
  Tag, 
  Key, 
  Compass, 
  FileText, 
  Sparkles,
  ArrowRight,
  RefreshCw
} from 'lucide-react';
import { NLPAPI } from '../services/api';

export default function NLPWorkbench() {
  const [activeSubTab, setActiveSubTab] = useState('ner');
  
  // NER & Keywords
  const [sampleText, setSampleText] = useState(
    'In this study, researchers at Stanford University implemented a Transformer architecture trained on the ImageNet dataset using PyTorch with AdamW optimizer and achieved 94.2% top-1 accuracy.'
  );
  const [entities, setEntities] = useState([]);
  const [keywords, setKeywords] = useState([]);
  const [extractLoading, setExtractLoading] = useState(false);

  // Semantic Similarity
  const [simTextA, setSimTextA] = useState('Convolutional neural networks extract hierarchical visual feature representations.');
  const [simTextB, setSimTextB] = useState('Vision transformers apply multi-head self-attention mechanisms to image patch tokens.');
  const [similarityScore, setSimilarityScore] = useState(null);
  const [simLoading, setSimLoading] = useState(false);

  // Summarization
  const [sumText, setSumText] = useState(
    'Deep learning models have revolutionized artificial intelligence across multiple domains including computer vision, natural language processing, and reinforcement learning. By utilizing multi-layered neural network architectures such as Convolutional Neural Networks, Recurrent Neural Networks, and Transformers, systems can automatically learn hierarchical representations directly from raw data without the need for manual feature engineering. Despite these advances, challenges such as computational complexity, high data requirements, and model interpretability remain active areas of ongoing research.'
  );
  const [summaryResult, setSummaryResult] = useState(null);
  const [sumLoading, setSumLoading] = useState(false);

  const handleExtractNLP = async () => {
    if (!sampleText.trim()) return;
    setExtractLoading(true);
    try {
      const [nerRes, kwRes] = await Promise.all([
        NLPAPI.extractEntities({ text: sampleText }),
        NLPAPI.extractKeywords({ text: sampleText, top_k: 8 }),
      ]);
      setEntities(nerRes.data.entities || []);
      setKeywords(kwRes.data.keywords || []);
    } catch (err) {
      console.error('NLP extraction failed', err);
    } finally {
      setExtractLoading(false);
    }
  };

  const handleComputeSimilarity = async () => {
    if (!simTextA.trim() || !simTextB.trim()) return;
    setSimLoading(true);
    try {
      const res = await NLPAPI.computeSimilarity({
        text_a: simTextA,
        text_b: simTextB,
        method: 'sentence_transformers',
      });
      setSimilarityScore(res.data);
    } catch (err) {
      console.error('Similarity calculation failed', err);
    } finally {
      setSimLoading(false);
    }
  };

  const handleSummarize = async () => {
    if (!sumText.trim()) return;
    setSumLoading(true);
    try {
      const res = await NLPAPI.summarizeText({
        text: sumText,
        max_length: 130,
        min_length: 30,
      });
      setSummaryResult(res.data);
    } catch (err) {
      console.error('Summarization failed', err);
    } finally {
      setSumLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
          <Search className="w-6 h-6 text-indigo-400" />
          NLP Intelligence & Transformer Workbench
        </h2>
        <p className="text-sm text-slate-400">
          Scientific Named Entity Recognition (NER), keyphrase ranking, cosine embedding similarity, and abstractive summarization.
        </p>
      </div>

      {/* Sub-tab navigation */}
      <div className="flex gap-2 border-b border-slate-800 pb-3">
        {[
          { id: 'ner', label: '1. NER & Keyphrase Extraction', icon: Tag },
          { id: 'similarity', label: '2. Cosine Semantic Similarity', icon: Compass },
          { id: 'summarize', label: '3. Transformer Summarization', icon: FileText },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition ${
                isActive
                  ? 'bg-indigo-600/30 text-indigo-200 border border-indigo-500/50 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/50'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab 1: NER & Keyphrase Extraction */}
      {activeSubTab === 'ner' && (
        <div className="space-y-6">
          <div className="glass-panel p-5 space-y-4">
            <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <Tag className="w-4 h-4 text-indigo-400" />
              Named Entity Recognition & Keyphrase Ranking
            </h3>

            <textarea
              rows={4}
              value={sampleText}
              onChange={(e) => setSampleText(e.target.value)}
              className="input-field font-mono text-xs"
              placeholder="Enter academic text for entity extraction..."
            />

            <button
              onClick={handleExtractNLP}
              disabled={extractLoading}
              className="btn-primary text-xs"
            >
              <Sparkles className={`w-3.5 h-3.5 ${extractLoading ? 'animate-spin' : ''}`} />
              {extractLoading ? 'Extracting Entities & Keyphrases...' : 'Run NER & Keyword Extraction'}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Extracted Named Entities */}
            <div className="glass-panel p-5 space-y-3">
              <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Tag className="w-4 h-4 text-cyan-400" />
                Detected Scientific Entities ({entities.length})
              </h4>
              {entities.length === 0 ? (
                <div className="text-xs text-slate-500 py-8 text-center">Click run to extract named entities.</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {entities.map((ent, idx) => (
                    <div key={idx} className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-xs flex items-center gap-2">
                      <span className="font-semibold text-slate-200">{ent.text}</span>
                      <span className="badge badge-cyan uppercase text-[10px]">{ent.label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Extracted Keyphrases */}
            <div className="glass-panel p-5 space-y-3">
              <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Key className="w-4 h-4 text-emerald-400" />
                Top Keyphrases & Importance Scores
              </h4>
              {keywords.length === 0 ? (
                <div className="text-xs text-slate-500 py-8 text-center">Click run to extract keyphrases.</div>
              ) : (
                <div className="space-y-2">
                  {keywords.map((kw, idx) => (
                    <div key={idx} className="flex justify-between items-center p-2 rounded-lg bg-slate-900 border border-slate-800 text-xs">
                      <span className="font-mono text-slate-300">{kw.keyword}</span>
                      <span className="badge badge-emerald font-mono">Score: {kw.score.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Semantic Similarity */}
      {activeSubTab === 'similarity' && (
        <div className="glass-panel p-5 space-y-5">
          <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
            <Compass className="w-4 h-4 text-cyan-400" />
            Sentence-Transformers Pairwise Cosine Similarity
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs text-slate-400 font-medium">Research Statement A:</label>
              <textarea
                rows={3}
                value={simTextA}
                onChange={(e) => setSimTextA(e.target.value)}
                className="input-field font-mono text-xs"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs text-slate-400 font-medium">Research Statement B:</label>
              <textarea
                rows={3}
                value={simTextB}
                onChange={(e) => setSimTextB(e.target.value)}
                className="input-field font-mono text-xs"
              />
            </div>
          </div>

          <button
            onClick={handleComputeSimilarity}
            disabled={simLoading}
            className="btn-primary text-xs"
          >
            <Compass className={`w-3.5 h-3.5 ${simLoading ? 'animate-spin' : ''}`} />
            {simLoading ? 'Calculating Cosine Distance...' : 'Compute Semantic Similarity'}
          </button>

          {similarityScore && (
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
              <div>
                <div className="text-xs text-slate-400">Embedding Model:</div>
                <div className="text-xs font-mono text-indigo-300">{similarityScore.method}</div>
              </div>

              <div className="text-right">
                <div className="text-xs text-slate-400">Cosine Similarity Score:</div>
                <div className="text-2xl font-bold text-emerald-400">
                  {(similarityScore.similarity_score * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Transformer Abstractive Summarization */}
      {activeSubTab === 'summarize' && (
        <div className="glass-panel p-5 space-y-4">
          <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
            <FileText className="w-4 h-4 text-purple-400" />
            Hugging Face Transformer Abstractive Summarization
          </h3>

          <div className="space-y-1.5">
            <label className="text-xs text-slate-400 font-medium">Source Document Text / Abstract:</label>
            <textarea
              rows={6}
              value={sumText}
              onChange={(e) => setSumText(e.target.value)}
              className="input-field font-mono text-xs"
            />
          </div>

          <button
            onClick={handleSummarize}
            disabled={sumLoading}
            className="btn-primary text-xs bg-gradient-to-r from-purple-600 to-indigo-600"
          >
            <Sparkles className={`w-3.5 h-3.5 ${sumLoading ? 'animate-spin' : ''}`} />
            {sumLoading ? 'Generating Abstractive Summary...' : 'Generate Academic Summary'}
          </button>

          {summaryResult && (
            <div className="p-4 rounded-xl bg-slate-900/90 border border-purple-800/40 space-y-2">
              <div className="flex justify-between items-center text-xs text-purple-300">
                <span className="font-semibold">Synthesized Summary:</span>
                <span className="font-mono">Reduction: {((1 - summaryResult.summary_words / summaryResult.original_words) * 100).toFixed(0)}%</span>
              </div>
              <p className="text-sm text-slate-200 leading-relaxed font-sans">
                {summaryResult.summary}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
