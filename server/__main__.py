from waitress import serve

from server import config
from server.app import app
from server.logger import logger


def main() -> None:
    logger.info('serving on 0.0.0.0:%s', config.PORT)
    serve(app, host='0.0.0.0', port=config.PORT)


if __name__ == '__main__':
    main()
