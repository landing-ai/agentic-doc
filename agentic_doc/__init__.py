import logging
import os
import sys

import structlog
from structlog.dev import better_traceback

_LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())

logging.basicConfig(
    format="%(message)s (%(filename)s:%(lineno)d)",
    stream=sys.stdout,
    level=_LOG_LEVEL,
    force=True,
)
# prevent logging during interpreter shutdown
logging.raiseExceptions = False


structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.dev.ConsoleRenderer(
            exception_formatter=better_traceback,
            colors=True,
            level_styles={
                "warning": structlog.dev.ConsoleRenderer.get_default_level_styles()[
                    "warning"
                ],
                "error": structlog.dev.ConsoleRenderer.get_default_level_styles()[
                    "error"
                ],
            },
        ),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(_LOG_LEVEL),
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=False,
)
