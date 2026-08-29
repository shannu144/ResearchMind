import React, { useState } from 'react';
import { 
  Share2, 
  Layers, 
  Play, 
  HelpCircle, 
  Tag, 
  Hash, 
  FolderTree,
  Sliders
} from 'lucide-react';
import { ClusteringAPI } from '../services/api';

export default function ClusteringViewer({ documents }) {
  const [algorithm, setAlgorithm] = useState('lda');
  const [nTopics, setNTopics] = useState(3);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleRunClustering = async () => {
    setLoading(true);
    setResults(null);
    try {
      if (algorithm === 'lda') {
        const res = await ClusteringAPI.lda({
          n_topics: parseInt(nTopics),
          max_features: 500,
          n_top_words: 8,
        });
        setResults(res.data);
      } else {
        const res = await ClusteringAPI.kmeans({
          n_clusters: parseInt(nTopics),
        });
        setResults(res.data);
      }
    } catch (err) {
      console.error('Clustering run failed', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
          <Share2 className="w-6 h-6 text-indigo-400" />
          Document Clustering & Topic Modeling Studio
        </h2>
        <p className="text-sm text-slate-400">
          Unsupervised thematic discovery across your research collection using Latent Dirichlet Allocation (LDA) and KMeans on TF-IDF vectors.
        </p>
      </div>

      <div className="glass-panel p-5 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="text-xs text-slate-400 font-medium block mb-1">Algorithm:</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setAlgorithm('lda')}
                className={`py-2 px-3 rounded-lg text-xs font-semibold border transition ${
                  algorithm === 'lda'
                    ? 'bg-indigo-600/30 border-indigo-500 text-indigo-200 shadow'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                LDA Topics
              </button>
              <button
                onClick={() => setAlgorithm('kmeans')}
                className={`py-2 px-3 rounded-lg text-xs font-semibold border transition ${
                  algorithm === 'kmeans'
                    ? 'bg-cyan-600/30 border-cyan-500 text-cyan-200 shadow'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                KMeans Clusters
              </button>
            </div>
          </div>

          <div>
            <label className="text-xs text-slate-400 font-medium block mb-1">
              Number of {algorithm === 'lda' ? 'Topics' : 'Clusters'}:
            </label>
            <select
              value={nTopics}
              onChange={(e) => setNTopics(e.target.value)}
              className="input-field text-xs"
            >
              <option value="2">2 {algorithm === 'lda' ? 'Topics' : 'Clusters'}</option>
              <option value="3">3 {algorithm === 'lda' ? 'Topics' : 'Clusters'}</option>
              <option value="4">4 {algorithm === 'lda' ? 'Topics' : 'Clusters'}</option>
              <option value="5">5 {algorithm === 'lda' ? 'Topics' : 'Clusters'}</option>
            </select>
          </div>

          <div className="flex items-end">
            <button
              onClick={handleRunClustering}
              disabled={loading}
              className="btn-primary w-full justify-center text-xs h-[38px]"
            >
              <Play className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Discovering Clusters...' : 'Run Topic Modeling'}
            </button>
          </div>
        </div>
      </div>

      {results && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <FolderTree className="w-4 h-4 text-cyan-400" />
              Discovered {algorithm === 'lda' ? 'Latent Topics' : 'Document Clusters'} ({results.topics?.length || results.clusters?.length})
            </h3>
            <span className="badge badge-cyan text-xs font-mono">{results.algorithm}</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {(results.topics || results.clusters || []).map((topic, idx) => (
              <div key={idx} className="glass-panel p-5 space-y-4 border-indigo-500/20 flex flex-col justify-between">
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <span className="badge badge-indigo text-[10px] uppercase font-mono">
                      {algorithm === 'lda' ? `Topic ${topic.topic_id}` : `Cluster ${topic.topic_id}`}
                    </span>
                    <span className="text-[11px] font-mono text-slate-500">
                      Coherence: {topic.coherence_score}
                    </span>
                  </div>

                  <h4 className="text-sm font-bold text-slate-100 leading-snug">
                    {topic.label}
                  </h4>

                  {/* Top words */}
                  <div className="space-y-1 pt-1">
                    <div className="text-[10px] text-slate-400 uppercase font-bold tracking-wider mb-1.5">
                      Top Distinctive Terms:
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {topic.top_words.map((tw, twIdx) => (
                        <span key={twIdx} className="badge badge-cyan text-[10px] font-mono">
                          {tw.word} ({tw.weight.toFixed(2)})
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Assigned documents */}
                <div className="pt-3 border-t border-slate-800 space-y-1 text-xs">
                  <div className="text-[10px] text-slate-400 uppercase font-bold">Assigned Documents:</div>
                  {topic.document_filenames && topic.document_filenames.length > 0 ? (
                    <div className="space-y-1">
                      {topic.document_filenames.map((fname, fIdx) => (
                        <div key={fIdx} className="text-slate-300 font-mono text-[11px] truncate">
                          📄 {fname}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-slate-500 text-[11px] italic">No dominant documents</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
