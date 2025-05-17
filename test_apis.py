import json
import requests
import sys
import matplotlib.pyplot as plt
import os
from datetime import datetime

def visualize_clusters(clusters, title):
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract cluster sizes
    cluster_ids = []
    sizes = []
    for cluster_id, data in clusters.items():
        cluster_ids.append(cluster_id)
        sizes.append(data['size'] if isinstance(data, dict) else len(data))
    
    # Create bar plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(cluster_ids, sizes)
    plt.title(f'Cluster Size Distribution - {title}', fontsize=14, pad=20)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Items', fontsize=12)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f'figures/cluster_distribution_{title.lower().replace(" ", "_")}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create pie chart for cluster size distribution
    plt.figure(figsize=(12, 8))
    plt.pie(sizes, labels=[f'Cluster {cid}' for cid in cluster_ids], autopct='%1.1f%%')
    plt.title(f'Cluster Size Distribution (Pie Chart) - {title}', fontsize=14, pad=20)
    
    # Save the pie chart
    filename = f'figures/cluster_distribution_pie_{title.lower().replace(" ", "_")}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return timestamp

def print_cluster_details(cluster_id, cluster_data, is_kirill_api=False):
    if is_kirill_api:
        print(f"\nCluster {cluster_id} ({len(cluster_data['packages'])} items):")
        print("Example errors:")
        for i, error in enumerate(cluster_data['packages'][:5]):  # Show up to 5 examples
            print(f"\n  Error {i+1}:")
            print("  " + "\n  ".join(error.split('\n')))  # Format multiline errors
            if i < 4:  # Add separator between errors, except for the last one
                print("  " + "-"*50)
    else:
        print(f"\nCluster {cluster_id} ({len(cluster_data)} items):")
        print("Examples:")
        for i, package in enumerate(cluster_data[:5]):  # Show up to 5 examples
            print(f"\n  Package {i+1}: {package['package']}")
            print(f"  Error Type: {package['error_type']}")
            print(f"  Language: {package['programming_language']}")
            print("  Description:")
            print("  " + "\n  ".join(package['description'].split('\n')))  # Format multiline descriptions
            if i < 4:  # Add separator between packages, except for the last one
                print("  " + "-"*50)

def print_language_stats(language_stats):
    print("\nLanguage Statistics:")
    print("-" * 60)
    print(f"{'Language':<20} {'Sample Count':<15} {'Clusters':<10}")
    print("-" * 60)
    for lang, stats in sorted(language_stats.items()):
        print(f"{lang:<20} {stats['count']:<15} {stats['clusters']:<10}")
    print("-" * 60)

def test_petr_api(data):
    print("\nTesting Petr's API:")
    
    # Test clustering endpoint
    response = requests.post("http://localhost:8000/petr/cluster", 
                           json={"data": data})
    if response.status_code == 200:
        print("Clustering successful")
        results = response.json()
        print(f"Number of packages clustered: {len(results)}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Test summary endpoint
    response = requests.post("http://localhost:8000/petr/cluster/summary", 
                           json={"data": data})
    if response.status_code == 200:
        print("\nDetailed Cluster Analysis:")
        summary = response.json()
        print(f"Total number of clusters: {summary['total_clusters']}")
        
        # Print language statistics
        print_language_stats(summary['language_stats'])
        
        # Visualize cluster distribution
        timestamp = visualize_clusters(summary['clusters'], "Petr's Clusters")
        print(f"\nVisualizations saved in figures directory (timestamp: {timestamp}):")
        print(f"- Bar chart: cluster_distribution_petrs_clusters_{timestamp}.png")
        print(f"- Pie chart: cluster_distribution_pie_petrs_clusters_{timestamp}.png")
        
        # Sort clusters by size
        sorted_clusters = sorted(
            summary['clusters'].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Print details for each cluster
        for cluster_id, cluster_data in sorted_clusters:
            print_cluster_details(cluster_id, cluster_data)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_kirill_api(data, n_clusters=10):
    print("\nTesting Kirill's API:")
    
    # Test clustering endpoint
    response = requests.post("http://localhost:8000/kirill/cluster", 
                           json={"data": data, "n_clusters": n_clusters})
    if response.status_code == 200:
        print("Clustering successful")
        results = response.json()
        print(f"Number of packages clustered: {len(results)}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Test summary endpoint
    response = requests.post("http://localhost:8000/kirill/cluster/summary", 
                           json={"data": data, "n_clusters": n_clusters})
    if response.status_code == 200:
        print("\nDetailed Cluster Analysis:")
        summary = response.json()
        print(f"Total number of clusters: {summary['total_clusters']}")
        
        # Visualize cluster distribution
        timestamp = visualize_clusters(summary['clusters'], "Kirill's Clusters")
        print(f"\nVisualizations saved in figures directory (timestamp: {timestamp}):")
        print(f"- Bar chart: cluster_distribution_kirills_clusters_{timestamp}.png")
        print(f"- Pie chart: cluster_distribution_pie_kirills_clusters_{timestamp}.png")
        
        # Sort clusters by size
        sorted_clusters = sorted(
            summary['clusters'].items(),
            key=lambda x: len(x[1]['packages']),
            reverse=True
        )
        
        # Print details for each cluster
        for cluster_id, cluster_data in sorted_clusters:
            print_cluster_details(cluster_id, cluster_data, is_kirill_api=True)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Load test data
    try:
        with open("petr.json", "r") as f:
            petr_data = json.load(f)
        print("\n=== Testing with Petr's data format ===")
        test_petr_api(petr_data)
    except FileNotFoundError:
        print("petr.json not found")
    except json.JSONDecodeError:
        print("Error parsing petr.json")

    try:
        with open("kirill.json", "r") as f:
            kirill_data = json.load(f)
        print("\n=== Testing with Kirill's data format ===")
        test_kirill_api(kirill_data)
    except FileNotFoundError:
        print("kirill.json not found")
    except json.JSONDecodeError:
        print("Error parsing kirill.json") 