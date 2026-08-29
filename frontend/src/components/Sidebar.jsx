import React from 'react';
import { 
  FileText, 
  BarChart2, 
  Cpu, 
  Search, 
  MessageSquare, 
  Sparkles, 
  Share2,
  FolderOpen
} from 'lucide-react';

const NAV_ITEMS = [
  { 
    id: 'documents', 
    label: 'Document Hub', 
    icon: FileText, 
    badge: 'Ingestion',
    desc: 'PDF, DOCX, CSV & FAISS' 
  },
  { 
    id: 'eda', 
    label: 'Corpus EDA', 
    icon: BarChart2, 
    badge: 'Analytics',
    desc: 'Vocabulary & N-Gram Stats' 
  },
  { 
    id: 'ml_lab', 
    label: 'ML & PyTorch Lab', 
    icon: Cpu, 
    badge: 'Models',
    desc: 'BiLSTM & Classifier Suite' 
  },
  { 
    id: 'nlp', 
    label: 'NLP Intelligence', 
    icon: Search, 
    badge: 'Inference',
    desc: 'NER, Keywords & Similarity' 
  },
  { 
    id: 'rag', 
    label: 'RAG Studio', 
    icon: MessageSquare, 
    badge: 'Hybrid RRF',
    desc: 'BM25 + FAISS & Triad Scorer' 
  },
  { 
    id: 'genai', 
    label: 'GenAI Science Suite', 
    icon: Sparkles, 
    badge: 'Synthesis',
    desc: 'Gaps, Review & BibTeX' 
  },
  { 
    id: 'clustering', 
    label: 'Topic Modeling', 
    icon: Share2, 
    badge: 'Unsupervised',
    desc: 'LDA & K-Means Clusters' 
  },
];

export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <aside className="w-64 border-r border-white/[0.07] bg-[#090a0f] p-3 flex flex-col justify-between shrink-0 min-h-[calc(100vh-53px)] select-none">
      <div className="space-y-1">
        <div className="px-3 py-2 text-[10px] font-bold tracking-wider text-slate-500 uppercase">
          Workspaces
        </div>

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left transition-all ${
                isActive
                  ? 'bg-[#151824] text-slate-100 font-medium border border-white/[0.08]'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-[#12141d] border border-transparent'
              }`}
            >
              <div className="flex items-center gap-2.5 min-w-0">
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-indigo-400' : 'text-slate-500'}`} />
                <span className="text-xs truncate">{item.label}</span>
              </div>
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                isActive ? 'bg-indigo-500/20 text-indigo-300 font-semibold' : 'bg-white/[0.04] text-slate-500'
              }`}>
                {item.badge}
              </span>
            </button>
          );
        })}
      </div>

      {/* Clean Engine Tech Stack Footer */}
      <div className="p-3 rounded-lg bg-[#0f111a] border border-white/[0.05] text-xs text-slate-400 space-y-1">
        <div className="flex items-center justify-between text-[11px] font-medium text-slate-300">
          <span>Engine Architecture</span>
          <span className="text-emerald-400 font-mono text-[10px]">PyTorch + FAISS</span>
        </div>
        <p className="text-[10px] text-slate-500 leading-normal">
          Hybrid BM25 + Dense vector search with quantitative RAG Triad evaluation.
        </p>
      </div>
    </aside>
  );
}
