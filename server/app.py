import signal
import sys
import threading

from flask import Flask, jsonify, request

from server import config
from server.algos import algos
from server.indexer import handle_event
from server.jetstream import run as run_jetstream
from server.logger import logger

app = Flask(__name__)

stream_stop_event = threading.Event()
stream_thread = threading.Thread(
    target=run_jetstream,
    args=(config.SERVICE_DID, handle_event, stream_stop_event),
    daemon=True,
)
stream_thread.start()


def _shutdown(*_):
    logger.info('stopping jetstream...')
    stream_stop_event.set()
    sys.exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


@app.get('/')
def index():
    return jsonify({
        'service': 'capital-region-feed',
        'did': config.SERVICE_DID,
        'feed': config.FEED_URI,
    })


@app.get('/healthz')
def healthz():
    return jsonify({'ok': True})


@app.get('/.well-known/did.json')
def did_json():
    if not config.SERVICE_DID.endswith(config.HOSTNAME):
        return '', 404
    return jsonify({
        '@context': ['https://www.w3.org/ns/did/v1'],
        'id': config.SERVICE_DID,
        'service': [
            {
                'id': '#bsky_fg',
                'type': 'BskyFeedGenerator',
                'serviceEndpoint': f'https://{config.HOSTNAME}',
            }
        ],
    })


@app.get('/xrpc/app.bsky.feed.describeFeedGenerator')
def describe_feed_generator():
    return jsonify({
        'did': config.SERVICE_DID,
        'feeds': [{'uri': uri} for uri in algos.keys()],
    })


@app.get('/xrpc/app.bsky.feed.getFeedSkeleton')
def get_feed_skeleton():
    feed = request.args.get('feed')
    algo = algos.get(feed)
    if not algo:
        return jsonify({'error': 'UnsupportedAlgorithm', 'message': 'Unsupported algorithm'}), 400

    try:
        cursor = request.args.get('cursor')
        limit = request.args.get('limit', default=20, type=int)
        body = algo(cursor, limit)
    except ValueError:
        return jsonify({'error': 'InvalidRequest', 'message': 'Malformed cursor'}), 400

    return jsonify(body)
