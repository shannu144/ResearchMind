import React, { useEffect, useState } from 'react';
import { Sparkles, Cpu, Database, RefreshCw, Layers, ShieldCheck } from 'lucide-react';
import { HealthAPI } from '../services/api';

export default function Navbar({ activeTab }) {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const res = await HealthAPI.detailed();
      setHealth(res.data);
    } catch (err) {
      console.log('Health check failed', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/[0.07] bg-[#090a0f]/95 backdrop-blur-md px-6 py-3 flex items-center justify-between">
      {/* Brand Identity */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm tracking-tighter shadow-sm">
          RM
        </div>
        <div className="flex items-center gap-2.5">
          <span className="text-sm font-semibold tracking-tight text-slate-100">
            ResearchMind
          </span>
          <span className="text-slate-600">/</span>
          <span className="text-xs text-slate-400 font-normal hidden sm:inline">
            Research Intelligence Suite
          </span>
          <span className="badge badge-slate text-[10px] uppercase font-mono font-bold tracking-wider py-0 px-1.5 ml-1">
            v2.0
          </span>
        </div>
      </div>

      {/* Clean System Metrics Status */}
      <div className="flex items-center gap-2.5 text-xs">
        {health ? (
          <>
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-[#151824] border border-white/[0.06] text-slate-300">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
              <span className="text-[11px] font-medium uppercase tracking-wider">{health.environment}</span>
            </div>

            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#151824] border border-white/[0.06] text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span className="text-[11px]">LLM: <strong className="text-slate-100 font-semibold">{health.llm_provider}</strong></span>
            </div>

            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#151824] border border-white/[0.06] text-slate-300">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-[11px]">FAISS: <strong className="text-slate-100 font-semibold">{health.indexed_vectors}</strong></span>
            </div>

            <button
              onClick={fetchHealth}
              title="Refresh System Health"
              className="p-1 rounded-md text-slate-400 hover:text-slate-200 hover:bg-[#1c2030] transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-indigo-400' : ''}`} />
            </button>
          </>
        ) : (
          <div className="flex items-center gap-1.5 text-slate-500 text-[11px]">
            <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-400" />
            <span>Connecting...</span>
          </div>
        )}
      </div>
    </header>
  );
}
