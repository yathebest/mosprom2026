"""Request-only access scope. Background inference uses job ownership, never a client tenant ID."""
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    organization_id: str
    cameras: tuple[str, ...] | None = None


current_scope: ContextVar[Scope | None] = ContextVar('safety_scope', default=None)


def sql_scope(table='jobs', camera_column=None):
    scope = current_scope.get()
    if scope is None:
        return '', []  # trusted internal workers / offline tools; HTTP always installs a scope
    sql, args = f' AND {table}.organization_id=?', [scope.organization_id]
    if scope.cameras is not None:
        column = camera_column or f"json_extract({table}.config,'$.camera')"
        sql += f" AND {column} IN ({','.join('?' for _ in scope.cameras) or 'NULL'})"
        args.extend(scope.cameras)
    return sql, args
