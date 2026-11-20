# Type stubs for the Flats Python SDK. The runtime is untyped ctypes; these
# describe what it accepts and returns.

from os import PathLike
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    overload,
)

__all__ = [
    "Database",
    "Result",
    "Cursor",
    "FlatsError",
    "InvalidArgumentError",
    "QueryError",
    "EngineError",
    "InternalError",
    "Interrupted",
    "version",
]

Vector = Sequence[float]
"""A list, tuple, ``array.array('f')``, NumPy array, … of numbers."""

Cell = Union[int, float, str, List[float]]
"""One value in a result: INT, FLOAT, TEXT, or a vector."""

Param = Union[int, float, str, Vector]
"""A value for one ``?``: INT, FLOAT (finite), TEXT, or a vector."""

Key = Union[int, str]
"""A primary key value: INT or TEXT."""

class FlatsError(Exception):
    code: int
    def __init__(self, message: str, code: Optional[int] = ...) -> None: ...

class InvalidArgumentError(FlatsError): ...
class QueryError(FlatsError): ...
class EngineError(FlatsError): ...
class InternalError(FlatsError): ...
class Interrupted(FlatsError): ...

def version() -> str: ...

class Result(Sequence[Tuple[Cell, ...]]):
    columns: Tuple[str, ...]
    @overload
    def __getitem__(self, i: int) -> Tuple[Cell, ...]: ...
    @overload
    def __getitem__(self, i: slice) -> Sequence[Tuple[Cell, ...]]: ...
    def __len__(self) -> int: ...
    def dicts(self) -> List[dict[str, Cell]]: ...

class Cursor:
    columns: Tuple[str, ...]
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...
    def __iter__(self) -> Iterator[Tuple[Cell, ...]]: ...
    def __next__(self) -> Tuple[Cell, ...]: ...
    def __enter__(self) -> "Cursor": ...
    def __exit__(self, *exc: Any) -> None: ...

class CollectionInfo(TypedDict):
    name: str
    dimension: int
    capacity: int
    rows: int
    primary_key: str

class ColumnInfo(TypedDict):
    name: str
    type: str
    """``"INT"``, ``"FLOAT"``, ``"TEXT"`` or ``"VECTOR(n)"``."""
    primary_key: bool

class CollectionDescription(CollectionInfo):
    columns: List[ColumnInfo]

Row = Mapping[str, Union[int, float, str, Vector]]
"""A row to write: every column by name, the vector included."""

class Database:
    BATCH_ROWS: int
    def __init__(self, path: Union[str, "PathLike[str]"], *, salvage: bool = ...) -> None: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Database": ...
    def __exit__(self, *exc: Any) -> None: ...
    def execute(self, sql: str, params: Sequence[Param] = ...) -> Result: ...
    def query(self, sql: str, params: Sequence[Param] = ..., *, chunk_rows: int = ...) -> Cursor: ...
    def search(
        self,
        collection: str,
        vector: Vector,
        k: int = ...,
        *,
        where: Optional[str] = ...,
        params: Sequence[Param] = ...,
        returning: Optional[Sequence[str]] = ...,
    ) -> Result: ...
    def insert(self, collection: str, rows: Iterable[Row]) -> int: ...
    def upsert(self, collection: str, rows: Iterable[Row]) -> int: ...
    def get(self, collection: str, key: Key) -> Optional[dict[str, Cell]]: ...
    def delete(self, collection: str, key: Key) -> None: ...
    def collections(self) -> List[CollectionInfo]: ...
    def describe(self, collection: str) -> CollectionDescription: ...
    def interrupt(self) -> None: ...
    def set_timeout(self, seconds: Optional[float]) -> None: ...
    def backup(self, dest: Union[str, "PathLike[str]"]) -> None: ...
