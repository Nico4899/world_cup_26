"""Smoke tests for the W1.4 auth + follow-teams SQLAlchemy models."""

from __future__ import annotations

from sqlalchemy import create_engine, inspect

from wc2026.db.models import (
    Base,
    SentKickoffPreview,
    User,
    UserFollowedTeam,
)


def test_user_model_columns() -> None:
    cols = {c.name for c in User.__table__.columns}
    assert cols == {
        "id",
        "name",
        "email",
        "emailVerified",  # NextAuth-canonical camelCase
        "image",
        "created_at",
        "last_login_at",
    }
    # `created_at` MUST have a SQL-side default — NextAuth's PG adapter
    # does raw INSERTs and will fail without one.
    assert User.__table__.c.created_at.server_default is not None


def test_user_followed_teams_composite_pk_and_cascade() -> None:
    cols = {c.name for c in UserFollowedTeam.__table__.columns}
    assert cols == {"user_id", "team", "created_at"}
    pk = [c.name for c in UserFollowedTeam.__table__.primary_key]
    assert sorted(pk) == ["team", "user_id"]
    # The FK to users.id must cascade-delete.
    fk = next(iter(UserFollowedTeam.__table__.c.user_id.foreign_keys))
    assert fk.ondelete == "CASCADE"


def test_sent_kickoff_previews_is_keyed_by_user_and_match() -> None:
    pk = [c.name for c in SentKickoffPreview.__table__.primary_key]
    assert sorted(pk) == ["match_id", "user_id"]


def test_all_three_new_tables_register_in_metadata() -> None:
    tables = set(Base.metadata.tables)
    for name in ("users", "user_followed_teams", "sent_kickoff_previews"):
        assert name in tables


def test_models_can_create_schema_on_sqlite() -> None:
    """``Base.metadata.create_all`` against an in-memory SQLite confirms
    the new models compose with the rest of the schema and the table
    layout is internally consistent."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    for name in ("users", "user_followed_teams", "sent_kickoff_previews"):
        assert name in insp.get_table_names()


def test_alembic_migration_module_loads() -> None:
    """The new migration file imports cleanly + declares the right ids.

    We don't actually run the migration here (that needs a real DB +
    Alembic env), but a syntax / import error in the migration would
    surface here. Load by path because ``alembic/versions`` isn't a
    Python package (no ``__init__.py``).
    """
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    mig_path = (
        repo_root / "alembic" / "versions" / "a4c0e1f2b3d4_users_followed_teams_auth.py"
    )
    spec = importlib.util.spec_from_file_location(
        "wc2026_test.migration_users", mig_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.revision == "a4c0e1f2b3d4"
    assert mod.down_revision == "f0d214c8a3b1"
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)
