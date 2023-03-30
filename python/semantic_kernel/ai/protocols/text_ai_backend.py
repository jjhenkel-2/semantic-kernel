# Copyright (c) Microsoft. All rights reserved.

from typing import Protocol, List, Tuple, Any


class TextAIBackend(Protocol):
    async def complete_single_async(self, prompt: str, **kwargs) -> str:
        """
        Creates a completion for the given prompt.

        Use kwargs to pass additional service-specific
        parameters to the completion backend.

        :param prompt: The prompt to complete.
        :param kwargs: Additional parameters to pass to the completion backend.

        :return: The completed prompt.
        """
        ...

    async def complete_multiple_async(self, prompt: str, n: int, **kwargs) -> List[str]:
        """
        Creates multiple completions for the given prompt.

        Use kwargs to pass additional service-specific
        parameters to the completion backend.

        NOTE: normally, you'll want to set the `temperature`
        parameter when generating multiple completion to a
        non-zero value. (Otherwise, the completions will be
        identical modulo some variance induced by floating
        point math.)

        :param prompt: The prompt to complete.
        :param n: The number of completions to generate.
        :param kwargs: Additional parameters to pass to the completion backend.

        :return: The completed prompts.
        """
        ...

    async def complete_single_with_metadata_async(
        self, prompt: str, logprobs: int, **kwargs
    ) -> Tuple[str, Any]:
        """
        Creates a completion for the given prompt and returns
        metadata about the completion (e.g., logprobs).

        Use kwargs to pass additional service-specific
        parameters to the completion backend.

        :param prompt: The prompt to complete.
        :param logprobs: Includes the log probabilities over
            the `logprobs` most likely tokens, as well as the chosen token.
        :param kwargs: Additional parameters to pass to the completion backend.

        :return: A tuple containing the completed prompt and
            metadata about the completion.
        """
        ...

    async def complete_multiple_with_metadata_async(
        self, prompt: str, n: int, logprobs: int, **kwargs
    ) -> List[Tuple[str, Any]]:
        """
        Creates multiple completions for the given prompt and
        returns metadata about the completions (e.g., logprobs).

        Use kwargs to pass additional service-specific
        parameters to the completion backend.

        NOTE: normally, you'll want to set the `temperature`
        parameter when generating multiple completion to a
        non-zero value. (Otherwise, the completions will be
        identical modulo some variance induced by floating
        point math.)

        :param prompt: The prompt to complete.
        :param n: The number of completions to generate.
        :param logprobs: Includes the log probabilities over
            the `logprobs` most likely tokens, as well as the chosen token.
        :param kwargs: Additional parameters to pass to the completion backend.

        :return: A list of tuples containing the completed
            prompts and metadata about the completions.
        """
        ...
