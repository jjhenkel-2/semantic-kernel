# Copyright (c) Microsoft. All rights reserved.

import openai

from typing import Tuple, List, Optional, Any, Type, Callable
from logging import Logger

from semantic_kernel.ai.protocols import ChatAIBackend
from semantic_kernel.utils.null_logger import NullLogger


class OpenAIChatBackend(ChatAIBackend):
    """
    An chat backend that uses the OpenAI Chat API
    to generate chat messages (based on a chat history).
    """

    # The list of parameters that are valid for the
    # when passed to generate_message_* methods via kwargs.
    # NOTE: we do not allow the `stream` parameter
    VALID_PARAMS = [
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "n",
        "logit_bias",
        "user",
    ]

    def __init__(
        self,
        model_id: str,
        api_key: str,
        org_id: Optional[str] = None,
        log: Optional[Logger] = None,
    ) -> None:
        """
        Initializes a new instance of the OpenAIChatBackend class.

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
        self._api_type = "standard"

        # Initialize the OpenAI module with the appropriate
        # API key and organization ID
        self._open_ai_instance = self._setup_open_ai()

        # A list of hooks to run on the chat_history before submission
        self._pre_chat_history_hooks: List[Callable[..., Any]] = []
        # A list of hooks to run on the response before returning it
        self._post_response_hooks: List[Callable[..., Any]] = []

    def _setup_open_ai(self) -> Type[openai.ChatCompletion]:
        """
        Sets up the OpenAI module with the appropriate
        API key and organization ID.

        :return: The OpenAI module with the appropriate
            API key and organization ID.
        """
        openai.api_key = self._api_key
        if self._org_id is not None:
            openai.organization = self._org_id

        return openai.ChatCompletion

    async def _generate_messages_async(
        self, chat_history: Tuple[str, str], **kwargs
    ) -> Any:
        # Do basic validation of the chat history
        if not chat_history:
            raise ValueError("chat_history cannot be empty")
        if chat_history[-1][0] != "user":
            raise ValueError("The last message in chat_history must be from the user")

        # Do basic validation of the parameters (just checking
        # to make sure we're passing in things that are supported)
        for param in kwargs:
            if param not in OpenAIChatBackend.VALID_PARAMS:
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
        if self._api_type == "azure":
            model_args["engine"] = self._model_id
        else:
            model_args["model"] = self._model_id

        # Run the pre-prompt hooks
        for hook in self._pre_chat_history_hooks:
            await hook(chat_history, **kwargs)

        # Finally, we'll generate a completion using the `openai.Completion` class
        formatted_messages = [
            {"role": role, "content": message} for role, message in chat_history
        ]
        response: Any = await self._open_ai_instance.acreate(
            messages=formatted_messages, **model_args
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
            "message" in choice for choice in response["choices"]
        ), "One or more of the choices did not contain a `message` field"
        # (4) each choice's message needs to have a `content` field
        assert all(
            "content" in choice["message"] for choice in response["choices"]
        ), "One or more of the choices's messages did not contain a `content` field"

        # Run the post-response hooks
        for hook in self._post_response_hooks:
            await hook(response, **kwargs)

        return response

    async def generate_message_async(
        self, chat_history: Tuple[str, str], **kwargs
    ) -> str:
        """
        Creates a message for the given chat history.

        Use kwargs to pass additional service-specific
        parameters to the completion backend.

        :param chat_history: The chat history to use to generate the message.
        :param kwargs: Additional parameters to pass to the chat completion backend.

        :return: A message generated from the chat history.
        """
        # Filter out the parameters that don't make sense for single completions
        unsupported_params = {"n"}
        kwargs = {
            param: value
            for param, value in kwargs.items()
            if param not in unsupported_params
        }

        response = await self._generate_messages_async(chat_history, **kwargs)
        return response["choices"][0]["message"]["content"]

    async def generate_messages_async(
        self, chat_history: Tuple[str, str], n: int, **kwargs
    ) -> List[str]:
        """
        Creates multiple messages for the given chat history.

        Use kwargs to pass additional service-specific
        parameters to the completion backend.

        NOTE: normally, you'll want to set the `temperature`
        parameter when generating multiple completion to a
        non-zero value. (Otherwise, the completions will be
        identical modulo some variance induced by floating
        point math.)

        :param chat_history: The chat history to use to generate the message.
        :param n: The number of messages to generate.
        :param kwargs: Additional parameters to pass to the chat completion backend.

        :return: The generated messages.
        """
        response = await self._generate_messages_async(chat_history, n=n, **kwargs)
        return [choice["message"]["content"] for choice in response["choices"]]
