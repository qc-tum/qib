import requests
import json

from qib.util import const


def _http_request(request, url: str, headers: dict, body: dict, title: str) -> requests.Response:
    """
    Generic HTTP Request function with retries.
    """
    log_title = f"[{title}] " if title else ""
    
    retries = 0
    while retries <= const.NW_MAX_RETRIES:
        try: # perform request
            response = request(url,
                               headers = headers,
                               json = body,
                               timeout = const.NW_TIMEOUT)
            try: # check response
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                raise RuntimeError(f"{log_title} HTTP error: {err}")
            except requests.exceptions.RequestException as err:
                raise RuntimeError(f"{log_title} Request error: {err}")
        except requests.exceptions.Timeout:
            retries += 1
            print(f"{log_title} Timeout error: {retries}/{const.NW_MAX_RETRIES} retries.")
            continue
        
        return response
    
    if retries > const.NW_MAX_RETRIES:
        raise RuntimeError(f"{log_title} Timeout error: Maximum retries reached.")


def http_put(url: str, headers: dict, body: dict, title: str = None) -> requests.Response:
    """
    Send a HTTP PUT request to the given URL with the given headers and json data.
    (Optionally) Specify a title for logging.
    """
    return _http_request(requests.put, url, headers, body, title)

def http_post(url: str, headers: dict, body: dict, title: str = None) -> requests.Response:
    """
    Send a HTTP POST request to the given URL with the given headers and json data.
    (Optionally) Specify a title for logging.
    """
    return _http_request(requests.post, url, headers, body, title)