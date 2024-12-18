import json
import string

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3 import Retry

from src.api.constants import (
    AFLOW_API,
    AFLOW_DEFAULT_PAGING,
    AFLOW_KEYWORDS,
    AFLOW_OPERATORS,
    AFLOW_SERVER,
    HTTP_PROTOCOLS,
    HTTP_STATUS_FORCELIST,
)
from src.typing import AfluxResponse


class AflowAPI:
    """A wrapper for the AFLOW API.
    This class provides a simple interface for querying the AFLOW API and retrieving data.
    The API documentation can be found at https://aflow.org/documentation/.

    Attributes:
        SERVER (str): The AFLOW server URL.
        API (str): The AFLOW API endpoint.
        PROTOCOLS (list[str]): The supported protocols for the API.
        STATUS_FORCELIST (list[int]): The list of HTTP status codes to force a retry.
        API_KEYWORDS (list[str]): The list of valid keywords for the API.
        API_OPERATORS (list[str]): The list of valid operators for the API.
        DEFAULT_PAGING (int): The default number of entries per page.

    Methods:
        __init__(self, max_retries: int | None = None) -> None:
            Initializes a new instance of the AflowAPI class.
        __enter__(self):
            Enters the context manager.
        __exit__(self, exc_type, exc_value, traceback):
            Exits the context manager.
        _create_session(self):
            Creates a new requests session with optional retry configuration.
        _make_request(self, url: str) -> requests.Response:
            Makes a GET request to the specified URL and handles error responses.
        _is_query_valid(self, query: str) -> bool:
            Checks if the query string is valid based on the API's rules.
        base_url(self) -> str:
            Returns the base URL for API requests.
        request(self, matchbook: str, paging: int | None = None, chunk_size: int | None = None, no_directives: bool = False) -> AfluxResponse:
            Sends a request to the AFLOW API and retrieves the response.
    """

    SERVER = AFLOW_SERVER
    API = AFLOW_API
    PROTOCOLS = HTTP_PROTOCOLS
    STATUS_FORCELIST = HTTP_STATUS_FORCELIST
    API_KEYWORDS = AFLOW_KEYWORDS
    API_OPERATORS = AFLOW_OPERATORS
    DEFAULT_PAGING = AFLOW_DEFAULT_PAGING

    def __init__(
        self,
        max_retries: int | None = 5,
    ) -> None:
        """
        Args:
            max_retries (int | None, optional): The maximum number of retries for HTTP requests. Defaults to 5.
        """
        self.max_retries = max_retries
        self.session = self._create_session()

    def __enter__(self):
        """Enters the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context manager."""
        self.session.close()

    def _create_session(self):
        """Creates a new requests session with optional retry configuration.

        Returns:
            requests.Session: The created session.
        """
        session = requests.Session()

        if self.max_retries is not None:
            retry = Retry(
                total=self.max_retries,
                read=self.max_retries,
                connect=self.max_retries,
                respect_retry_after_header=True,
                status_forcelist=self.STATUS_FORCELIST,
            )

            adapter = HTTPAdapter(max_retries=retry)
            for protocol in self.PROTOCOLS:
                session.mount(protocol, adapter)

        return session

    def _make_request(self, url: str) -> requests.Response:
        """Makes a GET request to the specified URL and handles error responses.

        Args:
            url (str): The URL to make the request to.

        Returns:
            requests.Response: The response object.

        Raises:
            RuntimeError: If the request fails with an HTTP error.
        """
        response = self.session.get(url)
        try:
            response.raise_for_status()
        except HTTPError as e:
            raise RuntimeError(f"Failed to download AFLUX data.\n\t{e}")

        return response

    def _is_query_valid(self, query: str) -> bool:
        """Checks if the query string is valid based on the API's rules.

        Args:
            query (str): The query string to validate.

        Returns:
            bool: True if the query is valid, False otherwise.
        """
        check_spaces = not any(c.isspace() for c in query)

        query_operators = [c for c in query if c in string.punctuation and c != "_"]
        check_operators = all(c in self.API_OPERATORS for c in query_operators)

        query_keywords = "".join([c for c in query if c.isalpha() or c == "_"])
        for key in self.API_KEYWORDS:
            if key in query_keywords:
                query_keywords = query_keywords.replace(key, "")
        check_keywords = len(query_keywords) == 0

        return check_spaces and check_operators and check_keywords

    @property
    def base_url(self) -> str:
        """Returns the base URL for API requests.

        Returns:
            str: The base URL.
        """
        return self.SERVER + self.API

    def request(
        self,
        matchbook: str,
        paging: int | None = None,
        chunk_size: int | None = None,
        no_directives: bool = False,
    ) -> AfluxResponse:
        """Sends a request to the AFLOW API and retrieves the response.

        Args:
            matchbook (str): The matchbook to query. See `https://aflow.org/documentation/` for more information.
            paging (int | None, optional): The page number for the request. By default, the query will be done on all pages at once. Defaults to None.
            chunk_size (int | None, optional): The number of entries per page. This number must be tuned if HttpError 500 happens. Defaults to None.
            no_directives (bool, optional): Whether to include directives in the request URL. Defaults to False.

        Returns:
            AfluxResponse: The response from AFLUX API in a JSON-like object.

        Raises:
            ValueError: If the chunk_size or paging values are invalid.
            ValueError: If the matchbook query is invalid.
        """
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be greater than 0")

        if paging is not None and paging < 0:
            raise ValueError("paging must be greater than or equal to 0")

        if not self._is_query_valid(matchbook):
            raise ValueError("Invalid query: contains invalid characters or keywords")

        paging = paging or self.DEFAULT_PAGING
        if chunk_size is not None:
            paging_str = f",$paging({paging},{chunk_size})"
        else:
            paging_str = f",$paging({paging})"

        request_url = self.base_url + matchbook
        if not no_directives:
            request_url = request_url + paging_str + "format(json)"

        response = self._make_request(request_url)

        try:
            json_response = response.json()
        except json.JSONDecodeError:
            raise RuntimeError("Failed to decode AFLUX response as JSON")

        return json_response
