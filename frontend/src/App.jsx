import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import DocumentManager from './components/DocumentManager';
import EDADashboard from './components/EDADashboard';
import MLDeepLearningLab from './components/MLDeepLearningLab';
import NLPWorkbench from './components/NLPWorkbench';
import RAGStudio from './components/RAGStudio';
import GenAISuite from './components/GenAISuite';
import ClusteringViewer from './components/ClusteringViewer';
import { DocumentAPI } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('documents');
  const [documents, setDocuments] = useState([]);

  const refreshDocuments = async () => {
    try {
      const res = await DocumentAPI.list();
      setDocuments(res.data);
    } catch (err) {
      console.error('Failed to sync documents in App', err);
    }
  };

  useEffect(() => {
    refreshDocuments();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#090a0f] text-slate-100 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Top Header */}
      <Navbar activeTab={activeTab} />

      {/* Full-width Responsive Body Layout */}
      <div className="flex-1 flex flex-col md:flex-row w-full min-h-[calc(100vh-53px)]">
        {/* Fixed Proportional Sidebar */}
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Full-Width Workspace Main Body */}
        <main className="flex-1 p-6 lg:p-8 overflow-y-auto w-full min-w-0 bg-[#090a0f]">
          <div className="w-full max-w-7xl mx-auto space-y-6">
            {activeTab === 'documents' && (
              <DocumentManager onDocumentsChanged={setDocuments} />
            )}

            {activeTab === 'eda' && (
              <EDADashboard documents={documents} />
            )}

            {activeTab === 'ml_lab' && (
              <MLDeepLearningLab />
            )}

            {activeTab === 'nlp' && (
              <NLPWorkbench />
            )}

            {activeTab === 'rag' && (
              <RAGStudio documents={documents} />
            )}

            {activeTab === 'genai' && (
              <GenAISuite documents={documents} />
            )}

            {activeTab === 'clustering' && (
              <ClusteringViewer documents={documents} />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
