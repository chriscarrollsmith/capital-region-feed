import os

import uvicorn

from server import config
from server.logger import logger


def main() -> None:
    logger.info('serving on 0.0.0.0:%s', config.PORT)
    uvicorn.run(
        'server.app:app',
        host='0.0.0.0',
        port=config.PORT,
        log_level=os.environ.get('LOG_LEVEL', 'INFO').lower(),
    )


if __name__ == '__main__':
    main()
