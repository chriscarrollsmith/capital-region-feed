import logging
import os

_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
_level = getattr(logging, _level_name, logging.INFO)

logging.basicConfig(
    level=_level,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)

logger = logging.getLogger('capital-region-feed')
