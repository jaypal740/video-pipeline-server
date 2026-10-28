from fastapi import APIRouter

from src.schema.api.response_schema import DataResponse, HealthView

router = APIRouter(tags=["health"])


@router.get("/health", response_model=DataResponse[HealthView])
def health() -> DataResponse[HealthView]:
    """Liveness check; needs no key."""
    return DataResponse(data=HealthView(status="ok"))
