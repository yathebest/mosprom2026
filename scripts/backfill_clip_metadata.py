"""Dry-run by default. --apply adds only verified clip metadata, with SQLite backup."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.clip_metadata import probe_clip,safe_path,cached_clip_metadata


def backfill(database,data_dir,*,apply=False):
    database=Path(database).resolve();data_dir=Path(data_dir).resolve()
    if not database.is_file():raise ValueError('Existing database file required')
    counts=Counter();plan=[]
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=20) as conn:
        conn.row_factory=sqlite3.Row
        for row in conn.execute("SELECT id,clip,payload FROM events WHERE clip IS NOT NULL AND clip<>''"):
            counts['scanned']+=1
            try:
                payload=json.loads(row['payload'] or '{}')
                if not isinstance(payload,dict):raise ValueError('Invalid event payload')
            except (ValueError,TypeError):counts['skipped_bad_payload']+=1;continue
            if payload.get('clip_state') not in (None,'ready'):
                counts['skipped_not_ready']+=1;continue
            try:path=safe_path(row['clip'],data_dir)
            except (OSError,ValueError):counts['skipped_invalid_path']+=1;continue
            if cached_clip_metadata(payload,path,data_dir=data_dir):counts['skipped_cached']+=1;continue
            metadata=probe_clip(path,data_dir=data_dir)
            if metadata is None:counts['skipped_unreadable']+=1;continue
            plan.append((row['id'],row['clip'],metadata))
        backup=None
        if apply and plan:
            folder=data_dir/'cache/clip-metadata-backups';folder.mkdir(parents=True,exist_ok=True)
            backup=folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.sqlite3')
            with sqlite3.connect(backup) as destination:conn.backup(destination)
    if apply and plan:
        with sqlite3.connect(database,timeout=20) as conn:
            conn.execute('BEGIN IMMEDIATE')
            for event_id,clip,metadata in plan:
                row=conn.execute('SELECT payload FROM events WHERE id=? AND clip=?',(event_id,clip)).fetchone()
                if not row:counts['skipped_changed']+=1;continue
                try:
                    payload=json.loads(row[0] or '{}')
                    if not isinstance(payload,dict) or payload.get('clip_state') not in (None,'ready'):raise ValueError()
                except (ValueError,TypeError):counts['skipped_changed']+=1;continue
                if not cached_clip_metadata(metadata,clip,data_dir=data_dir):counts['skipped_changed']+=1;continue
                if all(payload.get(key)==value for key,value in metadata.items()):counts['skipped_cached']+=1;continue
                # Fresh read under the write lock preserves concurrent review/lifecycle updates.
                payload.update(metadata)
                conn.execute('UPDATE events SET payload=? WHERE id=? AND clip=?',
                    (json.dumps(payload,ensure_ascii=False),event_id,clip));counts['updated']+=1
    return {'mode':'apply' if apply else 'dry_run','counts':dict(counts),'would_update':len(plan),
        'updated':counts['updated'],'backup':str(backup) if backup else None,
        'policy':'Only actual clip duration/file metadata is added. Source intervals, episode endpoints, statuses and audit rows are unchanged.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--database',type=Path);parser.add_argument('--apply',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=backfill(args.database or args.data_dir/'events.sqlite3',args.data_dir,apply=args.apply)
    text=json.dumps(result,indent=2);print(text)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(text,encoding='utf-8')
