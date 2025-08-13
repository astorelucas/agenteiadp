from pydantic import BaseModel
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from typing_extensions import Literal

class NormalizacaoSchema(BaseModel):
    columns: List[str] = Field(
        description="Lista de colunas numéricas a normalizar. Use '*' para todas as colunas numéricas."
    )
    method: Literal["minmax", "zscore"] = Field(
        description="Método de normalização. minmax -> escala entre [a,b], zscore -> média 0 e desvio 1."
    )
    feature_range: Optional[Tuple[float, float]] = Field(
        default=(0.0, 1.0),
        description="Faixa desejada para o método minmax. Ignorado para zscore."
    )
    imputar: bool = Field(
        default=False,
        description="Se True, preenche valores faltantes antes de normalizar."
    )
    imputacao_metodo: Literal["ffill", "bfill", "linear"] = Field(
        default="ffill",
        description="Método simples de imputação caso imputar=True."
    )