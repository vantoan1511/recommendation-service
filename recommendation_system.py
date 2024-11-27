import pandas as pd
from surprise import SVD, Dataset, Reader
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


# === Step 3: API Endpoint ===
@app.post("/evaluate", response_model=RecommendationResponse)
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

    # Step 2: Create an interaction DataFrame dynamically from behavior
    interactions = []
    for product in non_behavior_products:
        metadata_score = calculate_metadata_score(product, behavior, WEIGHTS)
        interactions.append([0, product.id, metadata_score])  # User ID is 0 (session-based)

    # Step 3: Train a lightweight collaborative filtering model
    interaction_df = pd.DataFrame(interactions, columns=["user_id", "product_id", "rating"])
    reader = Reader(rating_scale=(0, 5))
    surprise_data = Dataset.load_from_df(interaction_df, reader)
    trainset = surprise_data.build_full_trainset()
    model = SVD()
    model.fit(trainset)

    # Step 4: Score non-behavior products using collaborative filtering and metadata
    recommendations = []
    for product in non_behavior_products:
        predicted_rating = model.predict(0, product.id).est
        metadata_score = calculate_metadata_score(product, behavior, WEIGHTS)
        total_score = predicted_rating + metadata_score
        recommendations.append({
            "id": product.id,
            "name": product.name,
            "slug": product.slug,
            "basePrice": product.basePrice,
            "salePrice": product.salePrice,
            "stockQuantity": product.stockQuantity,
            "weight": product.weight,
            "color": product.color,
            "processor": product.processor,
            "gpu": product.gpu,
            "ram": product.ram,
            "storageType": product.storageType,
            "storageCapacity": product.storageCapacity,
            "os": product.os,
            "screenSize": product.screenSize,
            "batteryCapacity": product.batteryCapacity,
            "warranty": product.warranty,
            "model": {
                "id": product.model.id,
                "name": product.model.name,
                "slug": product.model.slug,
                "brand": {
                    "id": product.model.brand.id,
                    "name": product.model.brand.name,
                    "slug": product.model.brand.slug,
                },
            },
            "category": {
                "id": product.category.id,
                "name": product.category.name,
                "slug": product.category.slug,
            },
            "score": total_score,
        })

    recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)

    # Step 5: Append behavior products to the end with lower priority
    for product in behavior_products:
        metadata_score = calculate_metadata_score(product, behavior, WEIGHTS)
        recommendations.append({
            "id": product.id,
            "name": product.name,
            "slug": product.slug,
            "basePrice": product.basePrice,
            "salePrice": product.salePrice,
            "stockQuantity": product.stockQuantity,
            "weight": product.weight,
            "color": product.color,
            "processor": product.processor,
            "gpu": product.gpu,
            "ram": product.ram,
            "storageType": product.storageType,
            "storageCapacity": product.storageCapacity,
            "os": product.os,
            "screenSize": product.screenSize,
            "batteryCapacity": product.batteryCapacity,
            "warranty": product.warranty,
            "model": {
                "id": product.model.id,
                "name": product.model.name,
                "slug": product.model.slug,
                "brand": {
                    "id": product.model.brand.id,
                    "name": product.model.brand.name,
                    "slug": product.model.brand.slug,
                },
            },
            "category": {
                "id": product.category.id,
                "name": product.category.name,
                "slug": product.category.slug,
            },
            "score": metadata_score,  # Only metadata score for behavior products
        })




    return {"recommendations": recommendations}


# === Step 4: Run Server (For Local Testing) ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
