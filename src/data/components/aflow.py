from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from src.utils.typing import AfluxResponse, OptionalRange


class AflowAPI:
    server = "http://aflow.org"
    api = "/API/aflux/?"
    
    def __init__(
        self,
        max_retries: int = 3,
    ) -> None:
        self.max_retries = max_retries
        self.session = self._create_session()
        
    def _create_session(self):
        session = requests.Session()
        
        retry = Retry(
            total=self.max_retries,
            read=self.max_retries,
            connect=self.max_retries,
            respect_retry_after_header=True,
            status_forcelist=[429, 502, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
        
    def aflux_request(
        self,
        matchbook: str,
        paging: Optional[int] = None,
        paging_range: OptionalRange = None,
        no_directives: bool = False,
        retries: int = 0,
    ) -> AfluxResponse:
        """Download a AFLUX response and return it as list of dictionaries"""
        if paging is not None and paging_range is not None:
            raise ValueError("Cannot specify both paging and paging_range")

        request_url = self.server + self.api + matchbook

        if not no_directives:
            if paging is not None:
                request_url += f",$paging({paging}),format(json)"
            elif paging_range is not None:
                request_url += f",$paging({paging_range[0]},{paging_range[1]}),format(json)"
            else:
                request_url += ",$paging(0),format(json)"

        try:
            server_response = urlopen(request_url)
        except RemoteDisconnected:
            if retries > 0:
                print("RemoteDisconnected error, retrying...")
                return aflux_request(matchbook, paging, paging_range, no_directives, retries - 1)
            else:
                print("RemoteDisconnected error, no more retries left")
        else:
            response_content = server_response.read().decode("utf-8")

        # Basic error handling
        if server_response.getcode() == 200:
            try:
                return json.loads(response_content)
            except JSONDecodeError:
                pass

        print("AFLUX request failed!")
        print(f"  URL: {request_url}")
        print(f"  Response: {response_content}")
        return []

    def aflux_help(keyword: Optional[str] = None) -> None:
        """Print the build in help of AFLUX"""
        if keyword is None:
            # General help (https://aflow.org/API/aflux/?)
            help_data = aflux_request("", no_directives=True)
            print("\n".join(help_data))
        else:
            # Help regarding a specific keyword (https://aflow.org/API/aflux/?help(keyword))
            help_data = aflux_request(f"help({keyword})")
            for key, entry in help_data.items():
                print(key)
                print(f"  description: {entry['description']}")
                print(f"  units: {entry['units']}")
                print(f"  status: {entry['status']}")
                comment = "\n    ".join(entry["__comment__"]).strip()
                if comment:
                    print(f"  comment:\n    {comment}")
                    
    def aflux_get_contcar(entry: dict[str, str]) -> str | None:
        """Get a CONTCAR from AFLUX"""
        request_url = "http://" + \
            entry['aurl'].replace(':', '/') + '/' + 'CONTCAR.relax'
        server_response = urlopen(request_url)
        response_content = server_response.read().decode("utf-8")
        if server_response.getcode() == 200:
            # Fix POSTCAR if in VASP4 format
            poscar_lines = response_content.split('\n')
            # Add species names if missing
            if poscar_lines[5].strip()[0].isnumeric():
                poscar_lines.insert(5, " ".join(entry['species']))
            poscar = '\n'.join(poscar_lines)
            return poscar
        print("AFLUX request failed!")
        print(f"  URL: {request_url}")
        print(f"  Response: {response_content}")
        return None

    def aflux_get_property(entry: dict[str, str], property: str) -> float | None:
        """Get a property from AFLUX"""
        aurl = entry['aurl'].replace(':', '/')
        request_url = f"http://{aurl}/?{property}"
        server_response = urlopen(request_url)
        response_content = server_response.read().decode("utf-8")
        if server_response.getcode() == 200:
            # Fix POSTCAR if in VASP4 format
            poscar_lines = response_content.split('\n')
            # Add species names if missing
            if poscar_lines[5].strip()[0].isnumeric():
                poscar_lines.insert(5, " ".join(entry['species']))
            poscar = '\n'.join(poscar_lines)
            return poscar
        print("AFLUX request failed!")
        print(f"  URL: {request_url}")
        print(f"  Response: {response_content}")
        return None
    