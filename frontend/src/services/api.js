import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const DocumentAPI = {
  upload: (formData) => api.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  list: (skip = 0, limit = 100) => api.get(`/documents/?skip=${skip}&limit=${limit}`),
  get: (id) => api.get(`/documents/${id}`),
  delete: (id) => api.delete(`/documents/${id}`),
  getPages: (id) => api.get(`/documents/${id}/pages`),
  preprocess: (id) => api.post(`/documents/${id}/preprocess`),
};

export const DataScienceAPI = {
  corpusSummary: () => api.get('/data-science/corpus-summary'),
  eda: (documentId) => api.get(`/data-science/csv/${documentId}/eda`),
  ngramAnalysis: (topK = 20) => api.get(`/data-science/ngrams?top_k=${topK}`),
};

export const MLAPI = {
  trainClassifier: (params) => api.post('/ml/train', params),
  predict: (params) => api.post('/ml/predict', params),
  evaluate: () => api.get('/ml/evaluation'),
};

export const DeepLearningAPI = {
  trainBiLSTM: (params) => api.post('/deep-learning/train', params),
  predictBiLSTM: (params) => api.post('/deep-learning/predict', params),
  compareMLvsDL: (params) => api.get('/deep-learning/compare'),
};

export const NLPAPI = {
  extractEntities: (params) => api.post('/nlp/ner', params),
  extractKeywords: (params) => api.post('/nlp/keywords', params),
  computeSimilarity: (params) => api.post('/nlp/similarity', params),
  summarizeText: (params) => api.post('/nlp/summarize', params),
};

export const EmbeddingsAPI = {
  createEmbeddings: (params) => api.post('/embeddings/create', params),
  searchEmbeddings: (params) => api.post('/embeddings/search', params),
  getStats: () => api.get('/embeddings/stats'),
};

export const RAGAPI = {
  query: (params) => api.post('/rag/query', params),
  summarizeDocument: (params) => api.post('/rag/summarize', params),
};

export const GenAIAPI = {
  findGaps: (params) => api.post('/genai/research-gaps', params),
  compareDocuments: (params) => api.post('/genai/compare-documents', params),
  generateLiteratureReview: (params) => api.post('/genai/literature-review', params),
  getCitationNetwork: (similarityThreshold = 0.15) => 
    api.get(`/genai/citation-network?similarity_threshold=${similarityThreshold}`),
  generateHypotheses: (params) => api.post('/genai/hypotheses', params),
  exportBibtex: (params) => api.post('/genai/export-bibtex', params),
};

export const ClusteringAPI = {
  lda: (params) => api.post('/clustering/lda', params),
  kmeans: (params) => api.post('/clustering/kmeans', params),
};

export const HealthAPI = {
  check: () => api.get('/health'),
  detailed: () => api.get('/health/detailed'),
};

export default api;
