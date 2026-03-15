"""Launch the inference server."""

import os
import sys
import logging

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        # logger.info("#####launch_server#####")
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
