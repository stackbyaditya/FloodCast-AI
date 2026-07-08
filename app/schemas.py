from typing import Dict, Union

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: Dict[str, Union[float, int, str]]