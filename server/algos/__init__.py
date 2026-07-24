from collections.abc import Callable
from typing import Any

from server.algos import feed

FeedHandler = Callable[[str | None, int], dict[str, Any]]

algos: dict[str, FeedHandler] = {
    feed.uri: feed.handler,
}
