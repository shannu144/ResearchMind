import React, { useState } from 'react';
import { 
  Cpu, 
  Play, 
  Activity, 
  Award, 
  Layers, 
  CheckCircle2, 
  AlertCircle,
  HelpCircle,
  BarChart,
  GitCommit
} from 'lucide-react';
import { MLAPI, DeepLearningAPI } from '../services/api';

export default function MLDeepLearningLab() {
  // ML State
  const [mlModelType, setMlModelType] = useState('logistic_regression');
  const [mlLoading, setMlLoading] = useState(false);
  const [mlResults, setMlResults] = useState(null);

  // DL State
  const [epochs, setEpochs] = useState(5);
  const [hiddenDim, setHiddenDim] = useState(64);
  const [dlLoading, setDlLoading] = useState(false);
  const [dlResults, setDlResults] = useState(null);

  // Comparison State
  const [comparison, setComparison] = useState(null);
  const [compLoading, setCompLoading] = useState(false);

  // Live Inference Playground
  const [inferText, setInferText] = useState('The proposed neural architecture achieves state of the art results on benchmark datasets.');
  const [inferResult, setInferResult] = useState(null);
  const [inferLoading, setInferLoading] = useState(false);

  const handleTrainML = async () => {
    setMlLoading(true);
    setMlResults(null);
    try {
      const res = await MLAPI.trainClassifier({
        model_type: mlModelType,
        max_features: 1000,
        ngram_range: [1, 2],
        test_size: 0.2,
      });
      setMlResults(res.data);
    } catch (err) {
      console.error('ML train failed', err);
    } finally {
      setMlLoading(false);
    }
  };

  const handleTrainDL = async () => {
    setDlLoading(true);
    setDlResults(null);
    try {
      const res = await DeepLearningAPI.trainBiLSTM({
        epochs: parseInt(epochs),
        hidden_dim: parseInt(hiddenDim),
        batch_size: 16,
        learning_rate: 0.001,
      });
      setDlResults(res.data);
    } catch (err) {
      console.error('DL train failed', err);
    } finally {
      setDlLoading(false);
    }
  };

  const handleCompare = async () => {
    setCompLoading(true);
    try {
      const res = await DeepLearningAPI.compareMLvsDL({
        test_samples: 50,
      });
      setComparison(res.data);
    } catch (err) {
      console.error('Comparison failed', err);
    } finally {
      setCompLoading(false);
    }
  };

  const handleInfer = async () => {
    if (!inferText.trim()) return;
    setInferLoading(true);
    try {
      const res = await MLAPI.predict({ text: inferText });
      setInferResult(res.data);
    } catch (err) {
      console.error('Inference failed', err);
    } finally {
      setInferLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
          <Cpu className="w-6 h-6 text-indigo-400" />
          ML & Deep Learning Benchmark Lab
        </h2>
        <p className="text-sm text-slate-400">
          Train and benchmark classical NLP models (TF-IDF + LR/RF/SVM) against PyTorch Bidirectional LSTM neural networks.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Module 1: Classical Machine Learning Pipeline */}
        <div className="glass-panel p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <Award className="w-4 h-4 text-cyan-400" />
              1. Scikit-Learn TF-IDF Classifier
            </h3>
            <span className="badge badge-cyan">Classical ML</span>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-400 font-medium block mb-1.5">Algorithm Selection:</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'logistic_regression', label: 'Logistic Reg.' },
                  { id: 'random_forest', label: 'Random Forest' },
                  { id: 'svm', label: 'Linear SVM' },
                ].map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setMlModelType(item.id)}
                    className={`py-2 px-3 rounded-lg text-xs font-semibold border transition ${
                      mlModelType === item.id 
                        ? 'bg-indigo-600/30 border-indigo-500 text-indigo-200 shadow'
                        : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleTrainML}
              disabled={mlLoading}
              className="btn-primary w-full justify-center"
            >
              <Play className={`w-4 h-4 ${mlLoading ? 'animate-spin' : ''}`} />
              {mlLoading ? 'Training & Cross-Validating...' : 'Train Scikit-Learn Model'}
            </button>
          </div>

          {mlResults && (
            <div className="space-y-3 pt-2 border-t border-slate-800">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Accuracy</div>
                  <div className="text-sm font-bold text-emerald-400">{(mlResults.metrics.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">F1-Score</div>
                  <div className="text-sm font-bold text-cyan-400">{(mlResults.metrics.f1_score * 100).toFixed(1)}%</div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Precision</div>
                  <div className="text-sm font-bold text-indigo-400">{(mlResults.metrics.precision * 100).toFixed(1)}%</div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Recall</div>
                  <div className="text-sm font-bold text-purple-400">{(mlResults.metrics.recall * 100).toFixed(1)}%</div>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 font-mono text-[11px] text-slate-400 flex justify-between">
                <span>Training samples: {mlResults.training_samples}</span>
                <span>Train time: {mlResults.train_time_seconds.toFixed(3)}s</span>
              </div>
            </div>
          )}
        </div>

        {/* Module 2: PyTorch Deep Learning BiLSTM */}
        <div className="glass-panel p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-400" />
              2. PyTorch BiLSTM Neural Network
            </h3>
            <span className="badge badge-purple">Deep Learning</span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400 font-medium block mb-1">Epochs:</label>
              <input 
                type="number" 
                value={epochs} 
                onChange={(e) => setEpochs(e.target.value)}
                min="1" 
                max="20"
                className="input-field py-1.5 text-xs"
              />
            </div>

            <div>
              <label className="text-xs text-slate-400 font-medium block mb-1">Hidden Dimension:</label>
              <select 
                value={hiddenDim} 
                onChange={(e) => setHiddenDim(e.target.value)}
                className="input-field py-1.5 text-xs"
              >
                <option value="32">32 units</option>
                <option value="64">64 units</option>
                <option value="128">128 units</option>
              </select>
            </div>
          </div>

          <button
            onClick={handleTrainDL}
            disabled={dlLoading}
            className="btn-primary w-full justify-center bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 shadow-purple-500/25"
          >
            <Play className={`w-4 h-4 ${dlLoading ? 'animate-spin' : ''}`} />
            {dlLoading ? 'Executing PyTorch Training Loop...' : 'Train PyTorch BiLSTM'}
          </button>

          {dlResults && (
            <div className="space-y-3 pt-2 border-t border-slate-800">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Val Accuracy</div>
                  <div className="text-sm font-bold text-emerald-400">{(dlResults.metrics.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Val Loss</div>
                  <div className="text-sm font-bold text-amber-400">{dlResults.metrics.loss.toFixed(4)}</div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Vocab Size</div>
                  <div className="text-sm font-bold text-cyan-400">{dlResults.vocab_size}</div>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Total Epochs</div>
                  <div className="text-sm font-bold text-purple-400">{dlResults.history.length}</div>
                </div>
              </div>

              {/* Loss curve snippet */}
              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 text-[10px] font-mono text-slate-400 space-y-1">
                <div className="text-slate-300 font-semibold mb-1">Training Loss History per Epoch:</div>
                <div className="flex gap-2 overflow-x-auto py-1">
                  {dlResults.history.map((h) => (
                    <div key={h.epoch} className="px-2 py-1 rounded bg-slate-900 border border-slate-800 shrink-0">
                      Ep {h.epoch}: Loss {h.train_loss.toFixed(3)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Module 3: Live Inference Playground & ML vs DL Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-panel p-5 space-y-4">
          <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-emerald-400" />
            Live Text Classification Inference Playground
          </h3>

          <div className="space-y-3">
            <textarea
              rows={3}
              value={inferText}
              onChange={(e) => setInferText(e.target.value)}
              placeholder="Paste a research abstract or excerpt here..."
              className="input-field font-mono text-xs"
            />

            <button
              onClick={handleInfer}
              disabled={inferLoading}
              className="btn-secondary text-xs"
            >
              <GitCommit className="w-3.5 h-3.5" />
              {inferLoading ? 'Classifying...' : 'Classify Topic & Confidence'}
            </button>
          </div>

          {inferResult && (
            <div className="p-4 rounded-xl bg-slate-900/90 border border-indigo-500/30 flex items-center justify-between">
              <div>
                <div className="text-xs text-slate-400">Predicted Research Discipline:</div>
                <div className="text-base font-bold text-indigo-300 capitalize">{inferResult.predicted_category}</div>
              </div>
              <div className="text-right">
                <div className="text-xs text-slate-400">Confidence Score:</div>
                <div className="text-base font-bold text-emerald-400">{(inferResult.confidence * 100).toFixed(1)}%</div>
              </div>
            </div>
          )}
        </div>

        {/* Head-to-Head Comparison Button & Card */}
        <div className="glass-panel p-5 space-y-4 flex flex-col justify-between">
          <div className="space-y-2">
            <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <BarChart className="w-4 h-4 text-indigo-400" />
              ML vs DL Benchmark
            </h3>
            <p className="text-xs text-slate-400">
              Directly compares Scikit-Learn TF-IDF vs PyTorch BiLSTM on speed, throughput, and accuracy.
            </p>
          </div>

          {comparison && (
            <div className="space-y-2 text-xs">
              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex justify-between">
                <span className="text-slate-400">Classical ML:</span>
                <span className="text-cyan-300 font-bold">{comparison.classical_ml.model} ({(comparison.classical_ml.accuracy * 100).toFixed(0)}%)</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex justify-between">
                <span className="text-slate-400">Deep Learning:</span>
                <span className="text-purple-300 font-bold">BiLSTM ({(comparison.deep_learning.accuracy * 100).toFixed(0)}%)</span>
              </div>
              <div className="p-2 rounded bg-indigo-950/40 border border-indigo-800/40 text-[11px] text-indigo-300">
                ⚡ Winner: <strong>{comparison.winner}</strong> ({comparison.reason})
              </div>
            </div>
          )}

          <button
            onClick={handleCompare}
            disabled={compLoading}
            className="btn-secondary w-full justify-center text-xs"
          >
            {compLoading ? 'Evaluating benchmarks...' : 'Run Head-to-Head Benchmark'}
          </button>
        </div>
      </div>
    </div>
  );
}
