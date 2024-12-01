import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

# === Step 1: API Models ===
class Brand(BaseModel):
    id: int
    name: str
    slug: str

class Model(BaseModel):
    id: int
    name: str
    slug: str
    brand: Brand

class Category(BaseModel):
    id: int
    name: str
    slug: str

class Product(BaseModel):
    id: int
    name: str
    slug: str
    basePrice: float
    salePrice: float
    stockQuantity: int
    weight: float
    color: str
    processor: str
    gpu: str
    ram: int
    storageType: str
    storageCapacity: int
    os: str
    screenSize: float
    batteryCapacity: float
    warranty: float
    model: Model
    category: Category
    rating: float  # Initial product rating

class Behavior(BaseModel):
    favourites: List[int]
    inCart: List[int]
    recentVisits: List[int]

class RecommendationRequest(BaseModel):
    availableProducts: List[Product]
    behavior: Behavior

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]


# === Step 2: Helper Functions ===

def calculate_metadata_score(product: Product, behavior: Behavior, weights: Dict[str, float]) -> float:
    """
    Calculate a score for the product based on metadata matching behavior.
    """
    score = 0.0
    if product.id in behavior.favourites:
        score += weights["favourites"]
    if product.id in behavior.inCart:
        score += weights["inCart"]
    if product.id in behavior.recentVisits:
        score += weights["recentVisits"]
    return score


def is_product_in_behavior(product_id: int, behavior: Behavior) -> bool:
    """
    Check if a product exists in any of the behavior categories.
    """
    return (
        product_id in behavior.favourites or
        product_id in behavior.inCart or
        product_id in behavior.recentVisits
    )


def calculate_content_scores(available_products: List[Product]) -> Dict[int, List[float]]:
    """
    Calculate content-based similarity scores for products using cosine similarity.
    """
    # Combine product metadata into a single text field
    product_features = [
        f"{p.category.name} {p.model.brand.name} {p.color} {p.processor} {p.gpu}"
        for p in available_products
    ]

    # Vectorize product features
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(product_features)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(feature_matrix)

    # Create a mapping of product ID to its similarity scores
    content_scores = {product.id: similarity_matrix[idx] for idx, product in enumerate(available_products)}

    return content_scores


# === Step 3: API Endpoint ===

@app.post("/api/recommendations", response_model=RecommendationResponse)
def evaluate(request: RecommendationRequest):
    """
    Evaluate user behavior and recommend products from the provided available products.
    """
    available_products = request.availableProducts
    behavior = request.behavior

    if not available_products:
        return {"recommendations": []}  # No products to recommend

    # Assign weights to behavior types (can be fine-tuned)
    WEIGHTS = {"favourites": 1.5, "inCart": 1.2, "recentVisits": 1.0}

    # Step 1: Separate products in behavior and products not in behavior
    behavior_products = [
        product for product in available_products if is_product_in_behavior(product.id, behavior)
    ]
    non_behavior_products = [
        product for product in available_products if not is_product_in_behavior(product.id, behavior)
    ]

    # Step 2: Collaborative Filtering Scores (using SVD)
    # Prepare interaction data for collaborative filtering
    interactions = []
    for product in non_behavior_products:
        metadata_score = calculate_metadata_score(product, behavior, WEIGHTS)
        interactions.append([0, product.id, metadata_score])  # User ID is 0 (session-based)

    # Train collaborative filtering model
    interaction_df = pd.DataFrame(interactions, columns=["user_id", "product_id", "rating"])
    reader = Reader(rating_scale=(0, 5))
    surprise_data = Dataset.load_from_df(interaction_df, reader)
    trainset = surprise_data.build_full_trainset()
    cf_model = SVD(random_state=42)
    cf_model.fit(trainset)

    collaborative_scores = {
        product.id: cf_model.predict(0, product.id).est for product in non_behavior_products
    }

    # Step 3: Content-Based Filtering Scores
    content_scores = calculate_content_scores(available_products)

    # Step 4: Combine Scores
    alpha = 0.6  # Weight for collaborative filtering; 1-alpha is for content-based filtering
    recommendations = []

    for product in non_behavior_products:
        cf_score = collaborative_scores.get(product.id, 0.0)
        content_score = content_scores[product.id][0]  # Self-similarity with itself
        final_score = alpha * cf_score + (1 - alpha) * content_score

        recommendations.append({
            "id": product.id,
            "score": final_score,
        })

    # Sort recommendations by score
    recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)

    # Step 5: Append behavior products to the end with lower priority
    for product in behavior_products:
        metadata_score = calculate_metadata_score(product, behavior, WEIGHTS)
        recommendations.append({
            "id": product.id,
            "score": metadata_score,  # Only metadata score for behavior products
        })

    return {"recommendations": recommendations}


# === Step 4: Run Server (For Local Testing) ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
