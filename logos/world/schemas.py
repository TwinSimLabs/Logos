from pydantic import Field
from typing import Dict, Any, List
from logos.core.schemas.base import Identity, BaseSchema

class Entity(Identity):
    """An object or concept in the Truth Graph."""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    domain: str
    category: str
    type: str

class Relation(BaseSchema):
    """A directed relationship type between entities."""
    predicate: str

class Fact(BaseSchema):
    """The atomic unit of truth (Subject-Predicate-Object)."""
    subject_id: str
    predicate: str
    object_id: str
    truth_value: float = Field(ge=0.0, le=1.0, default=1.0)
