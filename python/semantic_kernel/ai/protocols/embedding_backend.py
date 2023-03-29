# Copyright (c) Microsoft. All rights reserved.

from typing import Protocol, List
from numpy import ndarray


class EmbeddingBackend(Protocol):
    async def generate_embeddings_async(self, texts: List[str], **kwargs) -> ndarray:
        """
        Generates embeddings for the given texts.

        :param texts: The texts to generate embedding for.
        :param kwargs: Additional parameters to pass to the embedding backend.

        :return: The embeddings for the given texts.
        """
        ...
