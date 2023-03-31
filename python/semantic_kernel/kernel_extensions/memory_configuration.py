# Copyright (c) Microsoft. All rights reserved.

from typing import Optional

from semantic_kernel.ai import EmbeddingAIBackend
from semantic_kernel.kernel_base import KernelBase
from semantic_kernel.memory.protocols import MemoryStore
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory


def use_memory(
    kernel: KernelBase,
    storage: MemoryStore,
    embeddings_generator: Optional[EmbeddingAIBackend] = None,
) -> None:
    if embeddings_generator is None:
        factory = kernel.config.get_ai_backend(
            EmbeddingAIBackend, kernel.config.get_embedding_backend_service_id()
        )
        embeddings_generator = factory(kernel)

    if storage is None:
        raise ValueError("The storage instance provided is None")
    if embeddings_generator is None:
        raise ValueError("An embedding generator was not found/could not be created")

    kernel.register_memory(SemanticTextMemory(storage, embeddings_generator))
