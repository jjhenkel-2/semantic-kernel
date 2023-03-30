# Copyright (c) Microsoft. All rights reserved.

from typing import List, Optional, Protocol

from semantic_kernel.memory.memory_query_result import MemoryQueryResult


class SemanticTextMemory(Protocol):
    async def save_information_async(
        self,
        collection: str,
        text: str,
        id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Save some information into the semantic memory,
        keeping a copy of the source information

        :param collection: The collection to save the information into
        :param text: The text to save into the memory
        :param id: The unique id for this information
        :param description: An optional description for this information
        """
        ...

    async def save_reference_async(
        self,
        collection: str,
        text: str,
        external_id: str,
        external_source_name: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Save some information into the semantic memory,
        keeping only a reference to the source information

        :param collection: The collection to save the information into
        :param text: The text to save into the memory
        :param external_id: The unique id for the source information
            E.g., a URL or GUID to the original source
        :param external_source_name: The name of external source for
            this information, e.g., "MSTeams", "GitHub", "MyWebSite", etc.
        :param description: An optional description for this information
        """
        ...

    async def get_async(
        self,
        collection: str,
        query: str,
    ) -> Optional[MemoryQueryResult]:
        """
        Fetch information from the semantic memory by key
        For local memories, the key is the `id` used when saving the record
        For remote memories, the key is the `external_id` (URI/GUID/eec.)
        used when saving the record

        :param collection: The collection to search for the information
        :param query: The key (unique ID) to search for

        :return: The information, or None if not found
        """
        ...

    async def search_async(
        self,
        collection: str,
        query: str,
        limit: int = 1,
        min_relevance_score: float = 0.7,
    ) -> List[MemoryQueryResult]:
        """
        Find some information in the semantic memory
        Returns a list of results, sorted by relevance

        :param collection: The collection to search within
        :param query: The query to search for
        :param limit: The maximum number of results to return
        :param min_relevance_score: The minimum relevance score to return

        :return: A list of results, sorted by relevance
        """
        ...

    async def get_collections_async(self) -> List[str]:
        """
        Gets a list of all the collection names in
        this semantic memory

        :return: A list of collection names
        """
        ...
