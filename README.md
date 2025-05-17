# Package Error Clustering Service

This service provides an API for clustering package build errors using two different approaches:

1. Petr's approach: DBSCAN clustering grouped by programming language
2. Kirill's approach: K-means clustering on error text

## Features

### Petr's Clustering
- Groups packages by programming language first
- Uses TF-IDF and DBSCAN for semantic clustering of errors
- Considers both error type and description
- Automatically determines number of clusters

### Kirill's Clustering
- Uses K-means clustering with configurable number of clusters
- Applies TF-IDF vectorization on full error text
- Focuses on error similarity regardless of language
- Allows control over cluster count

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
python api.py
```

The service will start on http://localhost:8000

## API Endpoints

### Petr's Clustering

#### POST /petr/cluster
Clusters packages using DBSCAN grouped by programming language.

Request body:
```json
{
    "data": [
        {
            "package": "package-name",
            "error_type": "error-type",
            "programming_language": "language",
            "description": "error description"
        }
        ...
    ]
}
```

#### POST /petr/cluster/summary
Returns detailed information about clusters from Petr's approach.

### Kirill's Clustering

#### POST /kirill/cluster
Clusters packages using K-means.

Request body:
```json
{
    "data": [
        {
            "package": "package-name",
            "errors": "full error text"
        }
        ...
    ],
    "n_clusters": 10  // Optional, defaults to 10
}
```

#### POST /kirill/cluster/summary
Returns detailed information about clusters from Kirill's approach.

### GET /health
Health check endpoint.

## Example Usage

### Petr's Approach
```python
import requests

data = {
    "data": [
        {
            "package": "example-package",
            "error_type": "compilation_error",
            "programming_language": "C++",
            "description": "Error description"
        }
    ]
}

response = requests.post("http://localhost:8000/petr/cluster", json=data)
clusters = response.json()
```

### Kirill's Approach
```python
import requests

data = {
    "data": [
        {
            "package": "example-package",
            "errors": "Full error log text..."
        }
    ],
    "n_clusters": 5  # Optional
}

response = requests.post("http://localhost:8000/kirill/cluster", json=data)
clusters = response.json()
```

## Documentation

Once the service is running, visit http://localhost:8000/docs for the interactive API documentation. 