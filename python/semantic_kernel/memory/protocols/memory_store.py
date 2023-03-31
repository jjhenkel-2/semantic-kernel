# Copyright (c) Microsoft. All rights reserved.

from typing import Protocol

from semantic_kernel.ai import EmbeddingIndex
from semantic_kernel.memory.protocols.data_store import DataStore


class MemoryStore(DataStore, EmbeddingIndex, Protocol):
    """
    A MemoryStore is a DataStore that also supports embedding-based similarity queries
    (A DataStore + EmbeddingIndex)
    """
