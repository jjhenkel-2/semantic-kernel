# Copyright (c) Microsoft. All rights reserved.

from typing import Protocol, List, runtime_checkable
from numpy import ndarray

from semantic_kernel.ai.ai_backend import AIBackend


@runtime_checkable
class EmbeddingAIBackend(AIBackend, Protocol):
    async def generate_embeddings_async(self, texts: List[str], **kwargs) -> ndarray:
        """
        Generates embeddings for the given texts.

        :param texts: The texts to generate embedding for.
        :param kwargs: Additional parameters to pass to the embedding backend.

        :return: The embeddings for the given texts.
        """
        ...
