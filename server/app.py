import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

from server import config
from server.algos import algos
from server.indexer import handle_event
from server.jetstream import run as run_jetstream
from server.logger import logger

stream_stop_event = threading.Event()
stream_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global stream_thread
    stream_stop_event.clear()
    stream_thread = threading.Thread(
        target=run_jetstream,
        args=(config.SERVICE_DID, handle_event, stream_stop_event),
        daemon=True,
    )
    stream_thread.start()
    try:
        yield
    finally:
        logger.info('stopping jetstream...')
        stream_stop_event.set()


app = FastAPI(title='capital-region-feed', lifespan=lifespan)


@app.get('/')
def index() -> dict[str, str]:
    return {
        'service': 'capital-region-feed',
        'did': config.SERVICE_DID,
        'feed': config.FEED_URI,
    }


@app.get('/healthz')
def healthz() -> dict[str, bool]:
    return {'ok': True}


@app.get('/.well-known/did.json', response_model=None)
def did_json() -> dict[str, Any] | Response:
    if not config.SERVICE_DID.endswith(config.HOSTNAME):
        return Response(status_code=404)
    return {
        '@context': ['https://www.w3.org/ns/did/v1'],
        'id': config.SERVICE_DID,
        'service': [
            {
                'id': '#bsky_fg',
                'type': 'BskyFeedGenerator',
                'serviceEndpoint': f'https://{config.HOSTNAME}',
            }
        ],
    }


@app.get('/xrpc/app.bsky.feed.describeFeedGenerator')
def describe_feed_generator() -> dict[str, Any]:
    return {
        'did': config.SERVICE_DID,
        'feeds': [{'uri': uri} for uri in algos.keys()],
    }


@app.get('/xrpc/app.bsky.feed.getFeedSkeleton', response_model=None)
def get_feed_skeleton(
    feed: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=20),
) -> dict[str, Any] | JSONResponse:
    algo = algos.get(feed) if feed is not None else None
    if not algo:
        return JSONResponse(
            {'error': 'UnsupportedAlgorithm', 'message': 'Unsupported algorithm'},
            status_code=400,
        )

    try:
        body = algo(cursor, limit)
    except ValueError:
        return JSONResponse(
            {'error': 'InvalidRequest', 'message': 'Malformed cursor'},
            status_code=400,
        )

    return body
