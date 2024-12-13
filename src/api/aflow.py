import json
import re
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
from src.utils.typing import AfluxResponse


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
        help(self, keyword: str | None = None) -> None:
            Displays help information for the AFLOW API.
        get_contcar(self, entry: dict[str, str]) -> str:
            Retrieves the CONTCAR file for a given entry.
        get_property(self, entry: dict[str, str], property: str) -> list[str]:
            Retrieves a specific property for a given entry.
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

    def help(self, keyword: str | None = None) -> None:
        """Displays help information for the AFLOW API.

        Args:
            keyword (str | None, optional): The specific keyword to get help for. None will display general help. Defaults to None.

        Raises:
            ValueError: If the keyword query is invalid.
        """
        # General help (https://aflow.org/API/aflux/?)
        if keyword is None:
            help_data = self.request("", no_directives=True)
            help_str = "\n".join(help_data)

        # Help regarding a specific keyword (https://aflow.org/API/aflux/?help(keyword))
        else:
            if not self._is_query_valid(keyword):
                raise ValueError("Invalid query: contains invalid keywords")

            try:
                help_data = self.request(f"help({keyword})", no_directives=True)
            except RuntimeError:
                print(f"No help information found for keyword: {keyword}")
                return

            entry = help_data[keyword]
            help_str = f"{keyword}:\n"
            help_str += f"  description: {entry['description']}\n"
            help_str += f"  units: {entry['units']}\n"
            help_str += f"  status: {entry['status']}\n"

            comment = "\n    ".join(entry["__comment__"]).strip()
            if comment:
                help_str += f"  comment:\n    {comment}"

        print(help_str)

    def get_contcar(self, entry: dict[str, str]) -> str:
        """Retrieves the CONTCAR file for a given entry.

        Args:
            entry (dict[str, str]): The entry containing the 'aurl' key.

        Returns:
            str: The contents of the CONTCAR file.

        Raises:
            ValueError: If the entry is missing the 'aurl' key.
        """
        if "aurl" not in entry.keys():
            raise ValueError("Invalid entry: missing 'aurl' key.")

        aurl = entry["aurl"].replace(":", "/")
        request_url = f"http://{aurl}/CONTCAR.relax"

        response = self._make_request(request_url)

        # Fix POSTCAR if in VASP4 format
        poscar_lines = response.text.split("\n")

        # Apply the lattice scaling factor
        scaling_factor = float(poscar_lines[1])
        if scaling_factor != 1.0:
            poscar_lines[1] = "1.0"
            for i in range(2, 5):
                poscar_lines[i] = " ".join(
                    str(float(x) * scaling_factor) for x in poscar_lines[i].split()
                )

        # Add species names if missing
        if poscar_lines[5].strip()[0].isnumeric():
            species = re.findall("[A-Z][a-z]*", entry["compound"])
            poscar_lines.insert(5, " ".join(species))

        # Remove the selective dynamics tag if present
        if poscar_lines[7].startswith(("s", "S")):
            del poscar_lines[7]

        # Convert the coordinates to Direct if in Cartesian format
        # and remove the potential selective dynamics tag
        n_atoms = sum(int(x) for x in poscar_lines[6].split())
        if poscar_lines[7].strip() == "Cartesian":
            poscar_lines[7] = "Direct"
            for i in range(8, 8 + n_atoms):
                poscar_lines[i] = " ".join(
                    str(float(x) * scaling_factor) for x in poscar_lines[i].split()[:3]
                )

        # Remove the velocities if present
        if len(poscar_lines) > 8 + n_atoms:
            poscar_lines = poscar_lines[: 8 + n_atoms]
            poscar_lines.append("")  # Add an empty line at the end

        poscar = "\n".join(poscar_lines)
        return poscar

    def get_property(self, entry: dict[str, str], property: str) -> list[str]:
        """Retrieves a specific property for a given entry.

        Args:
            entry (dict[str, str]): The entry containing the 'aurl' key.
            property (str): The property to retrieve.

        Returns:
            list[str]: The values of the property.

        Raises:
            ValueError: If the entry is missing the 'aurl' key.
        """
        if "aurl" not in entry.keys():
            raise ValueError("Invalid entry: missing 'aurl' key.")

        aurl = entry["aurl"].replace(":", "/")
        request_url = f"http://{aurl}/?{property}"

        response = self._make_request(request_url)

        property_value = response.text.strip().split(";")
        for value in property_value:
            value = value.strip().split(",")

        return property_value
