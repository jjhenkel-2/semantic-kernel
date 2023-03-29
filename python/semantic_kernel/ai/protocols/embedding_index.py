# Copyright (c) Microsoft. All rights reserved.

from numpy import ndarray
from typing import Protocol, List, Tuple

from semantic_kernel.memory.memory_record import MemoryRecord


class EmbeddingIndex(Protocol):
    async def get_nearest_matches_async(
        self,
        collection_name: str,
        key_embedding: ndarray,
        limit: int,
        min_relevance_score: float,
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        Gets the nearest matches for the given embedding.

        :param collection_name: The name of the collection to search.
        :param key_embedding: The embedding to use as our search key.
        :param limit: The maximum number of results to return.
        :param min_relevance_score: The minimum relevance score an
            embedding must have to eligible to be returned.

        :return: The nearest matches for the given embedding.
        """
        ...
