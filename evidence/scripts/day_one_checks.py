#!/usr/bin/env python3
"""Report this node's HTTPS, CPU SVD, storage, and Xet availability."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import time

from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
import torch


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=os.environ.get('ZOO_DATA'))
    args = parser.parse_args(argv)
    if args.data is None:
        parser.error('--data or ZOO_DATA is required')
    token = os.environ.get('HF_TOKEN')
    if not token:
        parser.error('HF_TOKEN must be set in the environment')
    if not args.data.is_dir():
        parser.error('the data directory must already exist')

    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    headers = {'Authorization': f'Bearer {token}'}
    outbound_https = False
    xet_reachable = False
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        opener = build_opener(NoRedirect())
        for field, url in (
            ('https', endpoint),
            ('xet', endpoint + '/openai-community/gpt2/resolve/main/model.safetensors'),
        ):
            request = Request(url, headers=headers, method='HEAD')
            try:
                try:
                    response = opener.open(request, timeout=20)
                except HTTPError as error:
                    response = error
                with response:
                    if field == 'https':
                        outbound_https = response.code < 400
                    else:
                        xet_reachable = response.code < 400 and bool(
                            response.headers.get('x-xet-hash')
                        )
            except (URLError, OSError):
                # Exceptions may contain signed URLs; report booleans only.
                pass

    matrix = torch.randn(3584, 18944, dtype=torch.float32, device='cpu',
                         generator=torch.Generator().manual_seed(20261028))
    started = time.perf_counter()
    torch.linalg.svdvals(matrix)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        'host': socket.gethostname(),
        'outbound_https': outbound_https,
        'svdvals_seconds': elapsed,
        'data_path': str(args.data),
        'data_free_gb': shutil.disk_usage(args.data).free / 1_000_000_000,
        'xet_reachable': xet_reachable,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
