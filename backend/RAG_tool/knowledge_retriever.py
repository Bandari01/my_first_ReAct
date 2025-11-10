"""
Knowledge retriever moved under backend.RAG_tool
"""
import json
import os
from pathlib import Path
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, using basic retrieval")


class KnowledgeRetriever:
    def __init__(self, knowledge_base_path: str | None = None):
        if knowledge_base_path is None:
            knowledge_base_path = Path(__file__).resolve().parent / "knowledge_base"
        self.knowledge_base_path = Path(knowledge_base_path)
        self.documents = []
        self.vectorizer = None
        self.tfidf_matrix = None

        self._load_knowledge_base()
        if SKLEARN_AVAILABLE:
            self._build_retrieval_system()
        else:
            print("Using basic keyword-based retrieval")

    def _load_knowledge_base(self):
        knowledge_files = [
            "data_analysis_patterns.json",
            "time_series_forecasting.json",
            "kaggle_best_practices.json",
            "common_errors.json",
            "store_sales_specific.json"
        ]

        for filename in knowledge_files:
            file_path = self.knowledge_base_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.documents.extend(self._process_knowledge_file(data, filename))
                    print("Loaded knowledge file:", filename)
                except Exception as e:
                    print("Error loading", filename, ":", e)

    def _process_knowledge_file(self, data, source):
        documents = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    doc = {
                        'content': item.get('content', ''),
                        'title': item.get('title', 'Untitled'),
                        'type': item.get('type', 'general'),
                        'source': source,
                        'tags': item.get('tags', [])
                    }
                    documents.append(doc)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    doc = {
                        'content': value.get('content', ''),
                        'title': value.get('title', key),
                        'type': value.get('type', 'general'),
                        'source': source,
                        'tags': value.get('tags', [])
                    }
                    documents.append(doc)

        return documents

    def _build_retrieval_system(self):
        if not self.documents:
            return

        texts = []
        for doc in self.documents:
            text = doc['title'] + " " + doc['content'] + " " + " ".join(doc.get('tags', []))
            texts.append(text)

        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query, top_k=5, similarity_threshold=0.1):
        if not self.documents:
            return []

        if not SKLEARN_AVAILABLE:
            return self._basic_retrieve(query, top_k)

        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > similarity_threshold:
                    result = self.documents[idx].copy()
                    result['similarity'] = float(similarities[idx])
                    results.append(result)

            return results
        except Exception as e:
            print("Retrieval error, falling back to basic:", e)
            return self._basic_retrieve(query, top_k)

    def _basic_retrieve(self, query, top_k):
        query_terms = query.lower().split()
        scored_docs = []

        for doc in self.documents:
            score = 0
            text = (doc['title'] + " " + doc['content']).lower()
            for term in query_terms:
                if len(term) > 3 and term in text:
                    score += 1

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(reverse=True)
        results = []
        for score, doc in scored_docs[:top_k]:
            result = doc.copy()
            result['similarity'] = score / len(query_terms)
            results.append(result)

        return results
