"""Entry point: python -m handformer.viz [--port 8768]"""
from __future__ import annotations

import argparse
import logging

import uvicorn

from .runner import Runner
from .server import create_app

log = logging.getLogger("handformer.viz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-through visualizer for the hand-written arithmetic transformer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runner = Runner()
    app = create_app(runner)
    log.info("serving on http://%s:%d", args.host, args.port)
    # log_config=None: uvicorn's loggers propagate to the root config above instead of installing their own.
    uvicorn.run(app, host=args.host, port=args.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
