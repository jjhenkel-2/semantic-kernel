# Copyright (c) Microsoft. All rights reserved.

from typing import Protocol, List, Tuple


class ChatAIBackend(Protocol):
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
        ...

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
        ...
