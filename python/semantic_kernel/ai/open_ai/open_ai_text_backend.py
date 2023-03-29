# Copyright (c) Microsoft. All rights reserved.

import openai

from logging import Logger
from typing import Any, Optional, List, Tuple, Callable

from semantic_kernel.ai.protocols import TextCompletionBackend
from semantic_kernel.utils.null_logger import NullLogger


class OpenAITextBackend(TextCompletionBackend):
    """
    A text completion backend that uses the OpenAI API
    to generate text completions.
    """

    # The list of parameters that are valid for the
    # when passed to complete_* methods via kwargs.
    # NOTE: we do note allow the `stream` parameter
    VALID_PARAMS = [
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "logprobs",
        "n",
        "suffix",
        "logit_bias",
        "user",
        "best_of",
        "echo",
    ]

    def __init__(
        self,
        model_id: str,
        api_key: str,
        org_id: Optional[str] = None,
        log: Optional[Logger] = None,
    ) -> None:
        """
        Initializes a new instance of the OpenAITextCompletion class.

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

        # A list of hooks to run on the prompt before submission
        self._pre_prompt_hooks: List[Callable[..., Any]] = []
        # A list of hooks to run on the response before returning it
        self._post_response_hooks: List[Callable[..., Any]] = []

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

    async def _complete_async(self, prompt: str, **kwargs: Any) -> Any:
        # Do basic validation of the prompt string
        if prompt is None:
            raise ValueError("The prompt cannot be None")
        if len(prompt) <= 0:
            raise ValueError("The prompt cannot be empty")

        # Do basic validation of the parameters (just checking
        # to make sure we're passing in things that are supported)
        for param in kwargs:
            if param not in OpenAITextBackend.VALID_PARAMS:
                raise ValueError(f"Invalid parameter '{param}'")

        # If the user didn't specify a max_tokens parameter,
        # we'll default to 128 tokens
        if "max_tokens" not in kwargs:
            self._log.info(
                "The max_tokens parameter was not specified, "
                "defaulting to 128 tokens"
            )
            kwargs["max_tokens"] = 128

        # Now we'll take all of the user-specified parameters
        # and make sure we correctly set the engine/model
        # parameter based on the OpenAI API type
        model_args = {**kwargs}
        if self._open_ai_instance.api_type == "azure":
            model_args["engine"] = self._model_id
        else:
            model_args["model"] = self._model_id

        # Run the pre-prompt hooks
        for hook in self._pre_prompt_hooks:
            await hook(prompt, **kwargs)

        # Finally, we'll generate a completion using the `openai` module
        response: Any = await self._open_ai_instance.Completion.acreate(
            prompt=prompt, **model_args
        )

        # Do some basic validation of the response (these should never fail,
        # unless OpenAI changes their API in a significant way)
        # (1) needs to be a `choices` field
        assert "choices" in response, "The response did not contain a `choices` field"
        # (2) needs to be a list of `choices`
        assert (
            len(response["choices"]) > 0
        ), "The response did not contain any `choices`"
        # (3) each choice needs to have a `text` field
        assert all(
            "text" in choice for choice in response["choices"]
        ), "One or more of the choices did not contain a `text` field"

        # Run the post-response hooks
        for hook in self._post_response_hooks:
            await hook(prompt, response, **kwargs)

        return response

    async def complete_single_async(self, prompt: str, **kwargs) -> str:
        """
        Generates a single completion using the OpenAI API.

        :param prompt: The prompt to complete.
        :param kwargs: Additional parameters to pass to the completion backend.
            Some examples of parameters that can be passed in are:
                - max_tokens: The maximum number of tokens to generate.
                - temperature: The temperature to use when generating the completion.
                - top_p: Restricts to tokens with top_p probability mass.
                - frequency_penalty: Penalizes tokens based on their frequency.
                - presence_penalty: Penalizes tokens based on their presence.
                - stop: A list of sequences to stop the completion on.
            See https://platform.openai.com/docs/api-reference/completions/create
            for a full list of parameters.
        """
        # Filter out the parameters that don't make sense for single completions
        # or completions that don't return any metadata
        unsupported_params = {"n", "logprobs", "echo"}
        kwargs = {
            param: value
            for param, value in kwargs.items()
            if param not in unsupported_params
        }

        # Now we'll generate a single completion
        response = await self._complete_async(prompt, **kwargs)

        # Finally, we'll return the (first&only) completion
        return response["choices"][0]["text"]

    async def complete_single_with_metadata_async(
        self, prompt: str, logprobs: int, **kwargs
    ) -> Tuple[str, Any]:
        """
        Generates a single completion using the OpenAI API.

        :param prompt: The prompt to complete.
        :param logprobs: The number of logprobs to return.
        :param kwargs: Additional parameters to pass to the completion backend.
            Some examples of parameters that can be passed in are:
                - max_tokens: The maximum number of tokens to generate.
                - temperature: The temperature to use when generating the completion.
                - top_p: Restricts to tokens with top_p probability mass.
                - frequency_penalty: Penalizes tokens based on their frequency.
                - presence_penalty: Penalizes tokens based on their presence.
                - stop: A list of sequences to stop the completion on.
            See https://platform.openai.com/docs/api-reference/completions/create
            for a full list of parameters.
        """
        # Filter out the parameters that don't make sense for single completions
        unsupported_params = {"n"}
        kwargs = {
            param: value
            for param, value in kwargs.items()
            if param not in unsupported_params
        }

        # Now we'll generate a single completion
        response = await self._complete_async(prompt, logprobs=logprobs, **kwargs)

        # Finally, we'll return the (first&only) completion
        return response["choices"][0]["text"], response["choices"][0]

    async def complete_multiple_async(self, prompt: str, n: int, **kwargs) -> List[str]:
        """
        Generates multiple completions using the OpenAI API.

        :param prompt: The prompt to complete.
        :param n: The number of completions to generate.
        :param kwargs: Additional parameters to pass to the completion backend.
            Some examples of parameters that can be passed in are:
                - max_tokens: The maximum number of tokens to generate.
                - temperature: The temperature to use when generating the completion.
                - top_p: Restricts to tokens with top_p probability mass.
                - frequency_penalty: Penalizes tokens based on their frequency.
                - presence_penalty: Penalizes tokens based on their presence.
                - stop: A list of sequences to stop the completion on.
            See https://platform.openai.com/docs/api-reference/completions/create
            for a full list of parameters.
        """
        # Filter out the parameters that don't make sense for
        # completions that don't return any metadata
        unsupported_params = {"logprobs", "echo"}
        kwargs = {
            param: value
            for param, value in kwargs.items()
            if param not in unsupported_params
        }

        # Let's generate completions
        response = await self._complete_async(prompt, n=n, **kwargs)

        # And return each of them
        return [choice["text"] for choice in response["choices"]]

    async def complete_multiple_with_metadata_async(
        self, prompt: str, n: int, logprobs: int, **kwargs
    ) -> List[Tuple[str, Any]]:
        """
        Generates multiple completions using the OpenAI API.

        :param prompt: The prompt to complete.
        :param n: The number of completions to generate.
        :param logprobs: Includes the log probabilities over
            the `logprobs` most likely tokens, as well as the chosen token.
        :param kwargs: Additional parameters to pass to the completion backend.
            Some examples of parameters that can be passed in are:
                - max_tokens: The maximum number of tokens to generate.
                - temperature: The temperature to use when generating the completion.
                - top_p: Restricts to tokens with top_p probability mass.
                - frequency_penalty: Penalizes tokens based on their frequency.
                - presence_penalty: Penalizes tokens based on their presence.
                - stop: A list of sequences to stop the completion on.
            See https://platform.openai.com/docs/api-reference/completions/create
            for a full list of parameters.
        """
        # Let's generate completions
        response = await self._complete_async(prompt, n=n, logprobs=logprobs, **kwargs)

        # And return each of them (both the text and the full completion object
        # which will include things like the logprobs, the finish_reason, etc.)
        return [(choice["text"], choice) for choice in response["choices"]]
