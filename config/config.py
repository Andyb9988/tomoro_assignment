import os
from dataclasses import dataclass
from typing import (
    Any,
    List,
)


@dataclass
class DataItem:
    context: str
    id: str
    question: str
