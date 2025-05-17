from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import math

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

def get_optimal_clusters(n_samples: int, min_clusters: int = 1, max_clusters: int = 20) -> int:
    """Calculate optimal number of clusters using logarithmic scale."""
    if n_samples <= 1:
        return 1
    if n_samples <= min_clusters:
        return min_clusters
    n_clusters = int(math.log2(n_samples)) + 1
    return min(max(n_clusters, min_clusters), min(max_clusters, n_samples))

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
            
        # Calculate optimal number of clusters for this language group
        n_clusters = get_optimal_clusters(len(lang_packages))
        print(f"Language: {language}, Samples: {len(lang_packages)}, Clusters: {n_clusters}")
            
        # If we only have one sample, assign it to a single cluster
        if len(lang_packages) == 1:
            lang_packages[0].cluster_id = cluster_id_counter
            cluster_id_counter += 1
            continue
            
        # Create feature vectors from error types and descriptions
        texts = [f"{p.error_type} {p.description}" for p in lang_packages]
        
        # Convert text to TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        features = vectorizer.fit_transform(texts)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features.toarray())
        
        # Assign cluster IDs
        for package, label in zip(lang_packages, cluster_labels):
            package.cluster_id = cluster_id_counter + label
            
        cluster_id_counter += n_clusters
    
    return packages

def get_clustering_results(json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Load and process packages
    packages = load_packages(json_data)
    clustered_packages = cluster_packages(packages)
    
    # Format results and convert numpy types to Python native types
    results = []
    for package in clustered_packages:
        results.append({
            "package": package.name,
            "cluster_id": int(package.cluster_id),  # Convert numpy.int64 to Python int
            "programming_language": package.programming_language
        })
    
    return results

def get_cluster_summary(json_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    packages = load_packages(json_data)
    clustered_packages = cluster_packages(packages)
    
    # Group packages by cluster
    clusters = defaultdict(list)
    for package in clustered_packages:
        clusters[int(package.cluster_id)].append({  # Convert numpy.int64 to Python int
            "package": package.name,
            "error_type": package.error_type,
            "programming_language": package.programming_language,
            "description": package.description
        })
    
    # Count packages per language
    language_counts = defaultdict(int)
    for package in packages:
        language_counts[package.programming_language] += 1
    
    return {
        "total_clusters": len(clusters),
        "clusters": {str(k): v for k, v in clusters.items()},  # Convert keys to strings
        "language_stats": {
            lang: {
                "count": count,
                "clusters": get_optimal_clusters(count)
            }
            for lang, count in language_counts.items()
        }
    } 