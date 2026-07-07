##############################################################################
# config/logging_config.py
#
# PURPOSE:
#   Central logging setup for the entire MOSAIC project.
#   Every single module — every agent, every API router, every ingestion
#   file — imports this one function and calls it the same way.
#   This means every log line across the entire system looks identical.
#
# WHAT PROBLEM DOES THIS SOLVE:
#   Without centralised logging, every developer sets up logging
#   differently. Some use print(). Some use logging.basicConfig().
#   The result is inconsistent — some logs have timestamps, some don't.
#   Some show the module name, some don't.
#   Central logging solves this — one setup, consistent everywhere.
#
# WHAT EVERY LOG LINE LOOKS LIKE:
#   2024-03-15 14:32:11 | INFO     | ingestion.clinical_trials_client | Fetching studies...
#   │                      │          │                                   │
#   timestamp              log level  which file it came from             the message
#
# HOW TO USE IN ANY FILE:
#   from config.logging_config import setup_logging
#   logger = setup_logging(__name__)
#   logger.info("Something happened")
#   logger.warning("Something might be wrong")
#   logger.error("Something went wrong")
##############################################################################


import logging
# logging is Python's built-in logging library.
# It gives us log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
# We use INFO for normal operations, WARNING for non-fatal issues,
# ERROR for failures. DEBUG is too verbose for production.

import sys
# sys.stdout is the standard output stream — the terminal.
# We write all logs to stdout so Cloud Run captures them automatically.
# Cloud Run reads stdout and sends it to Google Cloud Logging.
# No log files needed — Cloud Run handles storage for us.


def setup_logging(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger for the given module.

    This function is called at the top of every Python file in MOSAIC.
    Each file gets its own logger, namespaced to that file's path.
    This means log lines show exactly which file generated them.

    Args:
        name: Always pass __name__ here.
              __name__ is a Python built-in that equals the current
              module's full path. For example:
              - In ingestion/clinical_trials_client.py → "ingestion.clinical_trials_client"
              - In agents/supervisor.py → "agents.supervisor"
              - In api/main.py → "api.main"
              This is how we know which file each log line came from.

    Returns:
        logging.Logger: A fully configured logger ready to use.
                        Call .info(), .warning(), .error() on it.
    """

    logger = logging.getLogger(name)
    # getLogger(name) either creates a new logger or returns the
    # existing one with that name. If you call setup_logging("agents.supervisor")
    # twice, you get back the same logger object — not two separate ones.
    # This prevents duplicate log lines from appearing.

    if logger.handlers:
        # If this logger already has handlers attached, it is already
        # configured. Return it immediately without adding more handlers.
        # Without this check, every time a module is imported it would
        # add another handler, and the same log line would print twice,
        # three times, or more — one for each handler attached.
        return logger

    logger.setLevel(logging.INFO)
    # Set the minimum log level to INFO.
    # This means:
    #   logger.debug("x")    → NOT printed (below INFO threshold)
    #   logger.info("x")     → printed
    #   logger.warning("x")  → printed
    #   logger.error("x")    → printed
    #   logger.critical("x") → printed
    # We use DEBUG=False because debug logs are extremely verbose
    # and would flood the terminal during a real analysis run.

    # ── HANDLER ───────────────────────────────────────────────
    handler = logging.StreamHandler(sys.stdout)
    # StreamHandler sends log lines to a stream.
    # sys.stdout means: send them to the terminal (standard output).
    # This is the right choice for Cloud Run because:
    #   - Cloud Run automatically captures everything written to stdout
    #   - It sends it to Google Cloud Logging
    #   - We can see it in the GCP Console logs tab in real time
    # If we wrote to a file instead, Cloud Run would never see the logs.

    # ── FORMATTER ─────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # fmt controls how each log line is formatted.
    # Let's break down each part:
    #
    # %(asctime)s     → the timestamp, formatted by datefmt below
    #                   Example: 2024-03-15 14:32:11
    #
    # %(levelname)-8s → the log level, left-aligned in 8 characters
    #                   -8s means: pad with spaces to fill 8 characters
    #                   This keeps all log lines vertically aligned:
    #                   INFO     | ...
    #                   WARNING  | ...
    #                   ERROR    | ...
    #
    # %(name)s        → the logger name, which is the module path
    #                   Example: ingestion.clinical_trials_client
    #                   This tells us exactly which file logged this line
    #
    # %(message)s     → the actual message you passed to logger.info()
    #                   Example: "Fetching studies for diabetes"
    #
    # datefmt="%Y-%m-%d %H:%M:%S" formats the timestamp as:
    #   2024-03-15 14:32:11
    # We exclude milliseconds — cleaner for reading in the terminal.

    handler.setFormatter(formatter)
    # Attach the formatter to the handler.
    # The handler now knows how to format each line before printing it.

    logger.addHandler(handler)
    # Attach the handler to the logger.
    # Now when you call logger.info("something"), it flows through:
    #   logger → handler → formatter → stdout → terminal

    logger.propagate = False
    # propagate controls whether log messages travel up to the
    # root logger after being handled here.
    # Setting it to False stops that propagation.
    # Without this, every log line would print TWICE:
    #   once from our handler (correct)
    #   once from Python's root logger (duplicate, wrong)
    # False = our logger handles it, stop here, do not propagate up.

    return logger
    # Return the fully configured logger.
    # The calling module stores it as a module-level variable:
    #   logger = setup_logging(__name__)
    # And uses it throughout the file:
    #   logger.info("Starting ingestion pipeline")