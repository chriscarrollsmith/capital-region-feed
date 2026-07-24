import os

# Keep config imports safe if a test pulls in server packages beyond matcher.
os.environ.setdefault('HOSTNAME', 'test.example')
os.environ.setdefault(
    'FEED_URI',
    'at://did:plc:xndplob7sicvv6balxdzh3jk/app.bsky.feed.generator/aaagkkw3yejuk',
)
os.environ.setdefault('DATABASE_PATH', ':memory:')
