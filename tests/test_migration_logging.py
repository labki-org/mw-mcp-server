"""Running migrations at startup must not disable the app's loggers."""

import logging
from unittest.mock import patch

from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext


def _load_env(cfg: AlembicConfig) -> None:
    """Execute migrations/env.py with migrations themselves stubbed out."""
    script = ScriptDirectory.from_config(cfg)
    with EnvironmentContext(cfg, script), \
            patch("alembic.context.is_offline_mode", return_value=True), \
            patch("alembic.context.configure"), \
            patch("alembic.context.begin_transaction"), \
            patch("alembic.context.run_migrations"):
        script.run_env()


def test_app_startup_migrations_keep_app_loggers_enabled():
    app_logger = logging.getLogger("mcp.app")
    app_logger.disabled = False

    cfg = AlembicConfig("alembic.ini")
    cfg.attributes["configure_logger"] = False
    _load_env(cfg)

    assert not app_logger.disabled


def test_alembic_cli_still_configures_logging():
    app_logger = logging.getLogger("mcp.app")
    # fileConfig rewrites global logging state; restore it for later tests.
    root = logging.getLogger()
    saved_root = (root.level, list(root.handlers))
    saved_disabled = {
        name: lg.disabled
        for name, lg in logging.Logger.manager.loggerDict.items()
        if isinstance(lg, logging.Logger)
    }
    try:
        _load_env(AlembicConfig("alembic.ini"))
        # fileConfig ran: it disables pre-existing loggers not named in alembic.ini.
        assert app_logger.disabled
    finally:
        root.setLevel(saved_root[0])
        root.handlers[:] = saved_root[1]
        for name, disabled in saved_disabled.items():
            logging.getLogger(name).disabled = disabled
