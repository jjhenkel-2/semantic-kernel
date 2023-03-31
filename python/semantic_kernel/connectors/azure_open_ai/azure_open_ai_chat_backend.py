# Copyright (c) Microsoft. All rights reserved.

import openai

from logging import Logger
from typing import Optional, Type

from semantic_kernel.connectors.open_ai import OpenAIChatBackend


class AzureOpenAIChatBackend(OpenAIChatBackend):
    """
    A chat backend that uses the Azure OpenAI API
    to generate chat messages.
    """

    def __init__(
        self,
        deployment_name: str,
        endpoint: str,
        api_key: str,
        api_version: str = "2023-03-15-preview",
        log: Optional[Logger] = None,
    ) -> None:
        """
        Initializes a new instance of the OpenAIChatBackend class.

        :param deployment_name: The name of the Azure deployment. This value
            will correspond to the custom name you chose for your deployment
            when you deployed a model. This value can be found under
            Resource Management > Deployments in the Azure portal or, alternatively,
            under Management > Deployments in Azure OpenAI Studio.
        :param endpoint: The endpoint of the Azure deployment. This value
            can be found in the Keys & Endpoint section when examining
            your resource from the Azure portal.
        :param api_key: The API key for the Azure deployment. This value can be
            found in the Keys & Endpoint section when examining your resource in
            the Azure portal. You can use either KEY1 or KEY2.
        :param api_version: The API version to use. (Optional)
            The default value is "2023-03-15-preview".
        :param log: The logger instance to use. (Optional)
        """
        if not deployment_name or not deployment_name.strip():
            raise ValueError("Azure deployment_name cannot be empty")
        if not api_key or not api_key.strip():
            raise ValueError("Azure api_key cannot be empty")
        if not endpoint or not endpoint.strip():
            raise ValueError("Azure endpoint cannot be empty")
        if not endpoint.startswith("https://"):
            raise ValueError("Azure endpoint must start with 'https://'")

        super().__init__(deployment_name, api_key, log=log)

        self._endpoint = endpoint
        self._api_version = api_version
        self._api_type = "azure"

    def _setup_open_ai(self) -> Type[openai.ChatCompletion]:
        """
        Sets up the OpenAI module with the appropriate
        API key, API base, and API version.

        :return: The OpenAI module with the appropriate
            API key, API base, and API version.
        """
        openai.api_type = "azure"
        openai.api_key = self._api_key
        openai.api_base = self._endpoint
        openai.api_version = self._api_version

        return openai.ChatCompletion
