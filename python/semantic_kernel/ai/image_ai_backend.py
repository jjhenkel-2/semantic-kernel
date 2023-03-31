# Copyright (c) Microsoft. All rights reserved.

from typing import Protocol, runtime_checkable

from semantic_kernel.ai.ai_backend import AIBackend


@runtime_checkable
class ImageAIBackend(AIBackend, Protocol):
    async def generate_image_async(self, description: str, **kwargs) -> str:
        """
        Generates an image for the given description.

        :param description: The description to generate an image for.
        :param kwargs: Additional parameters to pass to the image generation backend.

        :return: A string (url/base64 encoded contents/other ref types possible)
            representing the generated image.
        """
        ...
