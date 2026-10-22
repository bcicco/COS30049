"""Liveness check."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Health(BaseModel):
    status: str


@router.get("/health")
def health() -> Health:
    return Health(status="ok")
