from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import traceback
import logging
from petr_cluster_service import get_clustering_results as petr_get_clustering_results
from petr_cluster_service import get_cluster_summary as petr_get_cluster_summary
from kirill_cluster_service import get_clustering_results as kirill_get_clustering_results
from kirill_cluster_service import get_cluster_summary as kirill_get_cluster_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Package Error Clustering API",
    description="API for clustering package errors using two different approaches",
    version="1.0.0"
)

class PackageData(BaseModel):
    data: List[Dict[str, Any]]

class KirillPackageData(BaseModel):
    data: List[Dict[str, Any]]
    n_clusters: int = 10

class LanguageStats(BaseModel):
    count: int
    clusters: int

class PetrClusterSummary(BaseModel):
    total_clusters: int
    clusters: Dict[str, List[Dict[str, Any]]]
    language_stats: Dict[str, LanguageStats]

# Petr's clustering endpoints
@app.post("/petr/cluster", response_model=List[Dict[str, Any]], tags=["Petr's Clustering"])
async def petr_cluster_packages(package_data: PackageData):
    """
    Cluster packages using Petr's approach: K-means clustering grouped by programming language.
    The number of clusters per language is calculated using a logarithmic scale based on the number of samples.
    Expects input data with programming_language, error_type, and description fields.
    """
    try:
        logger.info("Processing Petr's clustering request")
        result = petr_get_clustering_results(package_data.data)
        logger.info(f"Successfully processed {len(result)} packages")
        return result
    except Exception as e:
        logger.error(f"Error in Petr's clustering: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/petr/cluster/summary", response_model=PetrClusterSummary, tags=["Petr's Clustering"])
async def petr_get_clusters_summary(package_data: PackageData):
    """
    Get a detailed summary of all clusters using Petr's approach.
    Includes statistics about the number of clusters per programming language.
    """
    try:
        logger.info("Processing Petr's clustering summary request")
        result = petr_get_cluster_summary(package_data.data)
        logger.info(f"Successfully generated summary with {result['total_clusters']} clusters")
        return result
    except Exception as e:
        logger.error(f"Error in Petr's clustering summary: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Kirill's clustering endpoints
@app.post("/kirill/cluster", response_model=List[Dict[str, Any]], tags=["Kirill's Clustering"])
async def kirill_cluster_packages(package_data: KirillPackageData):
    """
    Cluster packages using Kirill's approach: K-means clustering on error text.
    Expects input data with package name and errors fields.
    """
    try:
        logger.info("Processing Kirill's clustering request")
        result = kirill_get_clustering_results(package_data.data, package_data.n_clusters)
        logger.info(f"Successfully processed {len(result)} packages")
        return result
    except Exception as e:
        logger.error(f"Error in Kirill's clustering: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/kirill/cluster/summary", response_model=Dict[str, Any], tags=["Kirill's Clustering"])
async def kirill_get_clusters_summary(package_data: KirillPackageData):
    """
    Get a detailed summary of all clusters using Kirill's approach.
    """
    try:
        logger.info("Processing Kirill's clustering summary request")
        result = kirill_get_cluster_summary(package_data.data, package_data.n_clusters)
        logger.info(f"Successfully generated summary with {result['total_clusters']} clusters")
        return result
    except Exception as e:
        logger.error(f"Error in Kirill's clustering summary: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 