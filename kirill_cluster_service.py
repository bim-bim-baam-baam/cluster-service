from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class KirillPackage:
    name: str
    errors: str
    cluster_id: int = -1

def load_packages(json_data: List[Dict[str, Any]]) -> List[KirillPackage]:
    packages = []
    for item in json_data:
        packages.append(KirillPackage(
            name=item['package'],
            errors=item['errors']
        ))
    return packages

def cluster_packages(packages: List[KirillPackage], n_clusters: int = 10) -> List[KirillPackage]:
    if not packages:
        return []
        
    # Create feature vectors from error descriptions
    texts = [p.errors for p in packages]
    
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        strip_accents='unicode',
        token_pattern=r'\b\w+\b'
    )
    features = vectorizer.fit_transform(texts)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=min(n_clusters, len(packages)), random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # Assign cluster IDs
    for package, label in zip(packages, cluster_labels):
        package.cluster_id = label
    
    return packages

def get_clustering_results(json_data: List[Dict[str, Any]], n_clusters: int = 10) -> List[Dict[str, Any]]:
    packages = load_packages(json_data)
    clustered_packages = cluster_packages(packages, n_clusters)
    
    # Convert to native Python types
    results = []
    for package in clustered_packages:
        results.append({
            'package': package.name,
            'errors': package.errors,
            'cluster_id': int(package.cluster_id)  # Convert numpy.int32 to Python int
        })
    return results

def get_cluster_summary(json_data: List[Dict[str, Any]], n_clusters: int = 10) -> Dict[str, Any]:
    packages = load_packages(json_data)
    clustered_packages = cluster_packages(packages, n_clusters)
    
    clusters = defaultdict(list)
    for package in clustered_packages:
        clusters[int(package.cluster_id)].append(package)  # Convert numpy.int32 to Python int
    
    summary = {
        'total_clusters': len(clusters),
        'clusters': {}
    }
    
    for cluster_id, packages in clusters.items():
        summary['clusters'][str(cluster_id)] = {
            'size': len(packages),
            'packages': [p.errors for p in packages]  # Return errors instead of package names
        }
    
    return summary 