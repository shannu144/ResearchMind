import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  FileText, 
  Trash2, 
  Layers, 
  CheckCircle2, 
  Clock, 
  AlertCircle, 
  RefreshCw,
  Search,
  Database,
  Eye,
  FileCheck
} from 'lucide-react';
import { DocumentAPI, EmbeddingsAPI } from '../services/api';

export default function DocumentManager({ onDocumentsChanged }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docPages, setDocPages] = useState([]);
  const [actionMessage, setActionMessage] = useState(null);
  const [embeddingLoading, setEmbeddingLoading] = useState({});

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const res = await DocumentAPI.list();
      setDocuments(res.data);
      if (onDocumentsChanged) onDocumentsChanged(res.data);
    } catch (err) {
      console.error('Failed to load documents', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    setActionMessage(null);
    try {
      const res = await DocumentAPI.upload(formData);
      setActionMessage({ type: 'success', text: `Uploaded "${res.data.filename}" successfully!` });
      await fetchDocuments();
    } catch (err) {
      setActionMessage({ 
        type: 'error', 
        text: err.response?.data?.detail || 'Failed to upload document.' 
      });
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleViewPages = async (doc) => {
    setSelectedDoc(doc);
    try {
      const res = await DocumentAPI.getPages(doc.id);
      setDocPages(res.data);
    } catch (err) {
      console.error('Failed to load pages', err);
    }
  };

  const handleCreateEmbeddings = async (docId) => {
    setEmbeddingLoading(prev => ({ ...prev, [docId]: true }));
    try {
      const res = await EmbeddingsAPI.createEmbeddings({
        document_id: docId,
        chunk_size: 400,
        chunk_overlap: 50,
      });
      setActionMessage({
        type: 'success',
        text: `FAISS Indexing complete: Generated ${res.data.chunks_created} vector chunks!`,
      });
    } catch (err) {
      setActionMessage({
        type: 'error',
        text: err.response?.data?.detail || 'Failed to create embeddings.',
      });
    } finally {
      setEmbeddingLoading(prev => ({ ...prev, [docId]: false }));
    }
  };

  const handleDelete = async (docId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) return;
    try {
      await DocumentAPI.delete(docId);
      if (selectedDoc?.id === docId) setSelectedDoc(null);
      await fetchDocuments();
    } catch (err) {
      console.error('Delete failed', err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top action header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <FileText className="w-6 h-6 text-indigo-400" />
            Document Hub
          </h2>
          <p className="text-sm text-slate-400">
            Ingest research papers (PDF, DOCX, TXT, CSV), extract metadata, and build FAISS vector index.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button 
            onClick={fetchDocuments}
            className="btn-secondary"
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>

          <label className="btn-primary cursor-pointer">
            <Upload className={`w-4 h-4 ${uploading ? 'animate-bounce' : ''}`} />
            {uploading ? 'Processing File...' : 'Upload Document'}
            <input 
              type="file" 
              className="hidden" 
              onChange={handleFileUpload}
              accept=".pdf,.docx,.txt,.csv"
              disabled={uploading}
            />
          </label>
        </div>
      </div>

      {actionMessage && (
        <div className={`p-4 rounded-xl flex items-center gap-3 text-sm border ${
          actionMessage.type === 'success' 
            ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-300' 
            : 'bg-rose-950/40 border-rose-800/60 text-rose-300'
        }`}>
          {actionMessage.type === 'success' ? (
            <CheckCircle2 className="w-5 h-5 shrink-0" />
          ) : (
            <AlertCircle className="w-5 h-5 shrink-0" />
          )}
          <span>{actionMessage.text}</span>
        </div>
      )}

      {/* Grid Layout: Document List + Document Page Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Document Table / Cards */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass-panel p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                <Database className="w-4 h-4 text-indigo-400" />
                Ingested Corpus ({documents.length})
              </h3>
              <span className="text-xs text-slate-500 font-mono">SQLite + FAISS Backend</span>
            </div>

            {loading ? (
              <div className="py-12 text-center text-slate-400 space-y-2">
                <RefreshCw className="w-6 h-6 animate-spin mx-auto text-indigo-500" />
                <p className="text-sm">Fetching document records...</p>
              </div>
            ) : documents.length === 0 ? (
              <div className="py-12 text-center border-2 border-dashed border-slate-800 rounded-xl space-y-3">
                <FileText className="w-10 h-10 mx-auto text-slate-600" />
                <div className="text-sm font-medium text-slate-300">No research papers uploaded yet</div>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Upload a PDF, TXT, CSV, or DOCX paper above to start EDA, ML, NLP, and RAG analysis.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {documents.map((doc) => {
                  const isSelected = selectedDoc?.id === doc.id;
                  const isEmbedding = embeddingLoading[doc.id];
                  return (
                    <div 
                      key={doc.id}
                      className={`p-4 rounded-xl border transition-all ${
                        isSelected 
                          ? 'bg-slate-800/90 border-indigo-500 shadow-md shadow-indigo-500/10' 
                          : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-start gap-3 min-w-0">
                          <div className="p-2.5 rounded-lg bg-indigo-950/60 border border-indigo-800/40 text-indigo-400 shrink-0 mt-0.5">
                            <FileText className="w-5 h-5" />
                          </div>
                          <div className="min-w-0">
                            <div className="text-sm font-semibold text-white truncate">
                              {doc.filename}
                            </div>
                            <div className="flex flex-wrap items-center gap-2 mt-1 text-xs text-slate-400">
                              <span className="badge badge-indigo uppercase">{doc.file_type}</span>
                              <span>•</span>
                              <span>{doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'}</span>
                              <span>•</span>
                              <span>{(doc.word_count || 0).toLocaleString()} words</span>
                              <span>•</span>
                              <span>{((doc.file_size || 0) / 1024).toFixed(1)} KB</span>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5 shrink-0">
                          <button
                            onClick={() => handleViewPages(doc)}
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition"
                            title="Inspect Text Content"
                          >
                            <Eye className="w-4 h-4" />
                          </button>

                          <button
                            onClick={() => handleCreateEmbeddings(doc.id)}
                            disabled={isEmbedding}
                            className="p-1.5 rounded-lg bg-indigo-900/40 hover:bg-indigo-800/60 border border-indigo-700/50 text-indigo-300 hover:text-indigo-100 transition"
                            title="Generate FAISS Embeddings"
                          >
                            <Database className={`w-4 h-4 ${isEmbedding ? 'animate-spin' : ''}`} />
                          </button>

                          <button
                            onClick={() => handleDelete(doc.id)}
                            className="p-1.5 rounded-lg bg-rose-950/30 hover:bg-rose-900/50 border border-rose-800/40 text-rose-400 hover:text-rose-200 transition"
                            title="Delete Document"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right: Document Inspector & Page Viewer */}
        <div className="space-y-4">
          <div className="glass-panel p-5 h-full">
            <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2 mb-3">
              <Eye className="w-4 h-4 text-cyan-400" />
              Content Inspector
            </h3>

            {selectedDoc ? (
              <div className="space-y-4">
                <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs space-y-1.5">
                  <div className="font-semibold text-slate-200 truncate">{selectedDoc.filename}</div>
                  <div className="text-slate-400 flex justify-between">
                    <span>Pages: {selectedDoc.page_count}</span>
                    <span>Words: {selectedDoc.total_words?.toLocaleString()}</span>
                  </div>
                </div>

                <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
                  {docPages.length === 0 ? (
                    <div className="text-xs text-slate-500 py-6 text-center">Loading page text...</div>
                  ) : (
                    docPages.map((p) => (
                      <div key={p.id} className="p-3 rounded-lg bg-slate-900/40 border border-slate-800/80 text-xs space-y-1.5">
                        <div className="flex items-center justify-between font-mono text-[10px] text-indigo-400">
                          <span>PAGE {p.page_number}</span>
                          <span>{p.word_count} words</span>
                        </div>
                        <p className="text-slate-300 leading-relaxed font-mono text-[11px] whitespace-pre-wrap max-h-40 overflow-y-auto">
                          {p.raw_text}
                        </p>
                      </div>
                    ))
                  )}
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-slate-500 text-xs space-y-2">
                <FileCheck className="w-8 h-8 mx-auto text-slate-700" />
                <p>Click the eye icon on any paper to inspect its extracted text & page breakdowns.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
