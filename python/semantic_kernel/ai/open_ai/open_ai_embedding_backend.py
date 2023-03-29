# Copyright (c) Microsoft. All rights reserved.

import openai

from typing import List, Optional, Any
from logging import Logger
from numpy import ndarray, array

from semantic_kernel.ai.protocols import EmbeddingBackend
from semantic_kernel.utils.null_logger import NullLogger


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """
    An embedding generation backend that uses the OpenAI API
    to generate text embeddings (returned as numpy arrays).
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        org_id: Optional[str] = None,
        log: Optional[Logger] = None,
    ) -> None:
        """
        Initializes a new instance of the OpenAIEmbeddingBackend class.

        :param model_id: OpenAI model name, see
            https://platform.openai.com/docs/models
        :param api_key: OpenAI API key, see
            https://platform.openai.com/account/api-keys
        :param org_id: OpenAI organization ID. (Optional)
            This is usually optional unless your
            account belongs to multiple organizations.
        :param log: The logger instance to use. (Optional)
        """
        if not model_id or not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")

        self._model_id = model_id
        self._api_key = api_key
        self._org_id = org_id
        self._log = log or NullLogger()

        # Initialize the OpenAI module with the appropriate
        # API key and organization ID
        self._open_ai_instance = self._setup_open_ai()

    def _setup_open_ai(self) -> openai:
        """
        Sets up the OpenAI module with the appropriate
        API key and organization ID.

        :return: The OpenAI module with the appropriate
            API key and organization ID.
        """
        openai.api_key = self._api_key
        if self._org_id is not None:
            openai.organization = self._org_id

        return openai

    async def generate_embeddings_async(self, texts: List[str], **kwargs) -> ndarray:
        """
        Generates embeddings for the given texts.

        :param texts: The texts to generate embedding for.
        :param kwargs: Additional parameters to pass to the embedding backend.

        :return: The embeddings for the given texts.
        """
        model_args = {**kwargs}
        if self._open_ai_instance.api_type == "azure":
            model_args["engine"] = self._model_id
        else:
            model_args["model"] = self._model_id

        # Generate the embeddings
        response: Any = self._open_ai_instance.Completion.create(
            input=texts, **model_args
        )

        # Do some basic validation of the response (these should never fail,
        # unless OpenAI changes their API in a significant way)
        # (1) The response should contain a `data` field
        assert "data" in response, "Response does not contain the `data` field"
        # (2) The `data` field should be a list (with the same length as the input)
        assert len(response["data"]) == len(
            texts
        ), "Response does not contain an embedding for each text"
        # (3) Each item in the `data` array should contain an `embedding` field
        assert all(
            "embedding" in raw_embedding for raw_embedding in response["data"]
        ), (
            "One or more items in the `data` array are missing the `embedding` field "
            f"(expected {len(texts)}, got {len(response['data'])})"
        )

        # Extract the embeddings as numpy arrays
        embeddings = [
            array(raw_embedding["embedding"]) for raw_embedding in response["data"]
        ]

        # Return the embeddings as a numpy array
        return array(embeddings)
