from typing import List, Dict, Any
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Package:
    name: str
    error_type: str
    programming_language: str
    description: str
    cluster_id: int = -1

def load_packages(json_data: List[Dict[str, Any]]) -> List[Package]:
    packages = []
    for item in json_data:
        if not item.get('programming_language'):
            continue
        packages.append(Package(
            name=item['package'],
            error_type=item['error_type'],
            programming_language=item['programming_language'],
            description=item['description']
        ))
    return packages

def cluster_packages(packages: List[Package]) -> List[Package]:
    # Group packages by programming language
    language_groups = defaultdict(list)
    for package in packages:
        language_groups[package.programming_language].append(package)
    
    # Process each language group separately
    cluster_id_counter = 0
    for language, lang_packages in language_groups.items():
        if not lang_packages:
            continue
            
        # Create feature vectors from error types and descriptions
        texts = [f"{p.error_type} {p.description}" for p in lang_packages]
        
        # Convert text to TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        features = vectorizer.fit_transform(texts)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=0.7, min_samples=1)
        cluster_labels = clustering.fit_predict(features.toarray())
        
        # Assign cluster IDs
        for package, label in zip(lang_packages, cluster_labels):
            package.cluster_id = cluster_id_counter + label
            
        cluster_id_counter += len(set(cluster_labels))
    
    return packages

def get_clustering_results(json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Load and process packages
    packages = load_packages(json_data)
    clustered_packages = cluster_packages(packages)
    
    # Format results
    results = []
    for package in clustered_packages:
        results.append({
            "package": package.name,
            "cluster_id": package.cluster_id,
            "programming_language": package.programming_language
        })
    
    return results

def get_cluster_summary(json_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    packages = load_packages(json_data)
    clustered_packages = cluster_packages(packages)
    
    # Group packages by cluster
    clusters = defaultdict(list)
    for package in clustered_packages:
        clusters[package.cluster_id].append({
            "package": package.name,
            "error_type": package.error_type,
            "programming_language": package.programming_language,
            "description": package.description
        })
    
    return {
        "total_clusters": len(clusters),
        "clusters": dict(clusters)
    } 