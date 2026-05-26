"""users + followed teams + NextAuth tables (W1.4 / Wave 5 personalisation)

Revision ID: a4c0e1f2b3d4
Revises: f0d214c8a3b1
Create Date: 2026-05-26 18:30:00.000000

Adds six tables that together support magic-link auth + follow-teams:

* ``users``                — application + NextAuth users row (PK = id).
* ``accounts``             — NextAuth OAuth account links (unused by the
                             Email provider but the adapter's schema
                             expects it; kept empty in production).
* ``sessions``             — NextAuth session cookies → user mapping.
* ``verification_tokens``  — outstanding magic-link tokens (15 min TTL).
* ``user_followed_teams``  — composite PK (user_id, team).
* ``sent_kickoff_previews``— idempotency table for the pre-kickoff email
                             job — short-circuits on (user_id, match_id).

Column names that look CamelCase (``emailVerified``, ``userId``,
``sessionToken``, ``providerAccountId``, ``expires_at``) match the
schema expected by ``@auth/pg-adapter`` verbatim — DO NOT rename them.
The application-only columns (``created_at``, ``last_login_at``) carry
SQL-side defaults so NextAuth's raw INSERTs don't fail.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4c0e1f2b3d4"
down_revision: str | Sequence[str] | None = "f0d214c8a3b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- NextAuth + application user row ----------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=True),
        sa.Column("email", sa.String(length=256), nullable=False),
        sa.Column("emailVerified", sa.DateTime(timezone=True), nullable=True),
        sa.Column("image", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # --- NextAuth OAuth accounts (unused by magic-link, kept for schema parity) -
    op.create_table(
        "accounts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "userId",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("providerAccountId", sa.String(length=128), nullable=False),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("access_token", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.BigInteger(), nullable=True),
        sa.Column("token_type", sa.String(length=32), nullable=True),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column("id_token", sa.Text(), nullable=True),
        sa.Column("session_state", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "provider",
            "providerAccountId",
            name="uq_accounts_provider_account",
        ),
    )
    op.create_index("ix_accounts_user_id", "accounts", ["userId"])

    # --- NextAuth sessions -------------------------------------------------
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("sessionToken", sa.String(length=256), nullable=False),
        sa.Column(
            "userId",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("expires", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("sessionToken", name="uq_sessions_token"),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["userId"])

    # --- NextAuth magic-link tokens ---------------------------------------
    op.create_table(
        "verification_tokens",
        sa.Column("identifier", sa.String(length=256), nullable=False),
        sa.Column("token", sa.String(length=256), nullable=False),
        sa.Column("expires", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(
            "identifier",
            "token",
            name="pk_verification_tokens",
        ),
    )

    # --- Application: follow-teams ----------------------------------------
    op.create_table(
        "user_followed_teams",
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("team", sa.String(length=128), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_user_followed_teams_team",
        "user_followed_teams",
        ["team"],
    )

    # --- Idempotency for the pre-kickoff email previews job ----------------
    op.create_table(
        "sent_kickoff_previews",
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("match_id", sa.Integer(), primary_key=True),
        sa.Column(
            "sent_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("sent_kickoff_previews")
    op.drop_index("ix_user_followed_teams_team", table_name="user_followed_teams")
    op.drop_table("user_followed_teams")
    op.drop_table("verification_tokens")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
    op.drop_index("ix_accounts_user_id", table_name="accounts")
    op.drop_table("accounts")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
