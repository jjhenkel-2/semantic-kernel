# Copyright (c) Microsoft. All rights reserved.

from typing import Any, List, Optional, Protocol

from semantic_kernel.memory.storage.data_entry import DataEntry


class DataStore(Protocol):
    async def get_collections_async(self) -> List[str]:
        """
        Gets a list of all available collection names

        :return: A list of collection names
        """
        ...

    async def get_all_async(self, collection: str) -> List[Any]:
        """
        Gets all entries from the specified collection

        :param collection: The collection name
        :return: A list of all entries in the collection
        """
        ...

    async def get_async(self, collection: str, key: str) -> Optional[DataEntry]:
        """
        Gets the entry with the specified key from the specified collection

        :param collection: The collection name
        :param key: The entry key

        :return: The entry with the specified key, or None if not found
        """
        ...

    async def put_async(self, collection: str, value: Any) -> DataEntry:
        """
        Puts the specified value into the specified collection

        :param collection: The collection name
        :param value: The value to put into the collection

        :return: The entry that was created
        """
        ...

    async def remove_async(self, collection: str, key: str) -> None:
        """
        Removes the entry with the specified key from the specified collection

        :param collection: The collection name
        :param key: The entry key
        """
        ...

    # TODO: what, if anything, should we do with these?
    async def get_value_async(self, collection: str, key: str) -> Any:
        ...

    # TODO: what, if anything, should we do with these?
    async def put_value_async(self, collection: str, key: str, value: Any) -> None:
        ...
