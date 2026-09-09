import sqlite3
import pytest
from stockrl_app.errors import AppError
from stockrl_app.settings import AppSettings
from stockrl_app.models import JobRecord
from stockrl_app.storage.database import Database
from stockrl_app.storage import migrations

# Frozen pre-migration schema; independent of the current production DDL.
V1_DDL = ('CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)',
 'CREATE TABLE datasets(dataset_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,\n'
 '        snapshot_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL)',
 'CREATE TABLE experiments(experiment_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,\n'
 '        dataset_id TEXT REFERENCES datasets(dataset_id), kind TEXT NOT NULL CHECK(kind IN '
 "('train','replay')),\n"
 '        source_experiment_id TEXT REFERENCES experiments(experiment_id),\n'
 '        job_id TEXT UNIQUE REFERENCES jobs(job_id) DEFERRABLE INITIALLY DEFERRED,\n'
 '        integrity TEXT NOT NULL CHECK(integrity IN '
 "('pending','complete','partial','corrupt','unsupported')),\n"
 '        output_relative_path TEXT, legacy_fingerprint TEXT, payload_json TEXT NOT NULL,\n'
 '        UNIQUE(output_relative_path, legacy_fingerprint))',
 'CREATE TABLE jobs(job_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL UNIQUE REFERENCES '
 'experiments(experiment_id),\n'
 "        kind TEXT NOT NULL CHECK(kind IN ('train','replay')), status TEXT NOT NULL\n"
 '        CHECK(status IN '
 "('queued','running','cancelling','succeeded','failed','cancelled','interrupted')),\n"
 '        created_at TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0), '
 'owner_token TEXT,\n'
 '        cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),\n'
 '        payload_json TEXT NOT NULL)',
 'CREATE TABLE job_events(job_id TEXT NOT NULL REFERENCES jobs(job_id), seq INTEGER NOT NULL '
 'CHECK(seq>0),\n'
 '        occurred_at TEXT NOT NULL, event_type TEXT NOT NULL CHECK(event_type IN '
 "('state','phase','progress','error')),\n"
 '        payload_json TEXT NOT NULL, PRIMARY KEY(job_id,seq))',
 'CREATE TABLE artifacts(artifact_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES '
 'experiments(experiment_id),\n'
 "        root_id TEXT NOT NULL CHECK(root_id IN ('app','output')), relative_path TEXT NOT NULL,\n"
 '        kind TEXT NOT NULL, size INTEGER NOT NULL CHECK(size>=0), sha256 TEXT NOT NULL,\n'
 '        payload_json TEXT NOT NULL, UNIQUE(experiment_id,root_id,relative_path))',
 'CREATE TABLE idempotency_keys(key TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL,\n'
 '        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),\n'
 '        job_id TEXT NOT NULL REFERENCES jobs(job_id), created_at TEXT NOT NULL)',
 'CREATE TABLE worker_instances(owner_token TEXT PRIMARY KEY, started_at TEXT NOT NULL,\n'
 '        heartbeat_at TEXT NOT NULL, recovery_waiting INTEGER NOT NULL DEFAULT 0,\n'
 '        active_job_id TEXT REFERENCES jobs(job_id), pid INTEGER, payload_json TEXT NOT NULL)',
 'CREATE INDEX datasets_page ON datasets(created_at DESC,dataset_id DESC)',
 'CREATE INDEX experiments_page ON experiments(created_at DESC,experiment_id DESC)',
 'CREATE INDEX jobs_queue ON jobs(status,created_at,job_id)',
 'CREATE INDEX workers_heartbeat ON worker_instances(heartbeat_at DESC)')


def old_database(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute('PRAGMA journal_mode=WAL')
    for statement in V1_DDL:
        conn.execute(statement)
    conn.execute("INSERT INTO schema_migrations VALUES (1,'original')")
    conn.execute('PRAGMA user_version=1')
    conn.execute("INSERT INTO datasets VALUES ('d','now','hash',' legacy bytes ')")
    conn.execute("INSERT INTO experiments(experiment_id,created_at,dataset_id,kind,integrity,payload_json) VALUES ('e','now','d','train','complete','legacy payload')")
    conn.execute("INSERT INTO jobs(job_id,experiment_id,kind,status,created_at,payload_json) VALUES ('j','e','train','succeeded','now','job payload')")
    conn.execute("UPDATE experiments SET job_id='j'")
    return conn


def test_new_schema_supports_resume_jobs_and_one_active_attempt(tmp_path):
    db = migrations.initialize_database(tmp_path / 'new.db')
    with db.transaction() as c:
        c.execute("INSERT INTO experiments(experiment_id,created_at,kind,integrity,payload_json) VALUES ('r','now','research','pending','{}')")
        c.execute("INSERT INTO researches(research_id,created_at,status,protocol_fingerprint,execution_environment_fingerprint,payload_json) VALUES ('r','now','queued','p','env','{}')")
        for i in (1,2):
            c.execute("INSERT INTO jobs(job_id,experiment_id,kind,status,created_at,payload_json) VALUES (?,'r','research','queued','now','{}')", (f'j{i}',))
        c.execute("INSERT INTO research_attempts VALUES ('a','r','j1','running','now','{}')")
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO research_attempts VALUES ('b','r','j2','queued','now','{}')")
        c.execute("UPDATE research_attempts SET status='interrupted'")
        c.execute("INSERT INTO research_attempts VALUES ('b','r','j2','queued','now','{}')")
        assert c.execute('PRAGMA foreign_key_check').fetchall() == []


def test_explicit_migration_keeps_wal_payload_and_backup(tmp_path):
    path, backup = tmp_path/'old.db', tmp_path/'backup.db'
    old = old_database(path)
    before = old.execute('SELECT * FROM experiments').fetchall()
    with pytest.raises(AppError):
        migrations.initialize_database(path)
    migrations.migrate_database(path, backup)
    with Database(path).connection() as c:
        assert tuple(c.execute('SELECT * FROM experiments').fetchone()) == before[0]
        assert c.execute('PRAGMA foreign_key_check').fetchall() == []
    with sqlite3.connect(backup) as c:
        assert c.execute('PRAGMA user_version').fetchone()[0] == 1
        assert c.execute('SELECT payload_json FROM datasets').fetchone()[0] == ' legacy bytes '
    old.close()


def test_failed_migration_rolls_back_and_retains_backup(tmp_path, monkeypatch):
    path, backup = tmp_path/'old.db', tmp_path/'backup.db'
    old = old_database(path)
    monkeypatch.setattr(migrations, 'V2_DDL', ('CREATE TABLE partial_change(x)', 'INVALID SQL'), raising=False)
    with pytest.raises(sqlite3.DatabaseError):
        migrations.migrate_database(path, backup)
    assert old.execute('PRAGMA user_version').fetchone()[0] == 1
    assert old.execute("SELECT name FROM sqlite_master WHERE name='partial_change'").fetchone() is None
    assert old.execute('SELECT payload_json FROM jobs').fetchone()[0] == 'job payload'
    assert backup.is_file()
    old.close()


def test_migration_rejects_active_writer_without_creating_backup(tmp_path):
    path, backup = tmp_path/'old.db', tmp_path/'backup.db'
    old = old_database(path)
    old.execute('BEGIN IMMEDIATE')
    with pytest.raises(AppError) as caught:
        migrations.migrate_database(path, backup)
    assert caught.value.code == 'MIGRATION_BUSY'
    assert not backup.exists()
    old.rollback()
    old.close()


def test_research_settings_and_jobs_remain_lightweight(tmp_path, monkeypatch):
    monkeypatch.setenv('STOCKRL_RESEARCH_DIR', 'safe/research')
    settings = AppSettings.from_env(tmp_path)
    assert settings.research_dir == tmp_path/'safe'/'research'
    assert settings.research_staging_dir == tmp_path/'safe'/'research'/'.staging'
    assert JobRecord(job_id='j', experiment_id='r', kind='research', created_at='now').public().kind == 'research'


def test_offline_lock_excludes_connections_and_initializer(tmp_path):
    path = tmp_path/'current.db'
    migrations.initialize_database(path)
    lock = path.with_name(path.name + '.migration.lock')
    lock.write_text('maintenance', encoding='utf-8')
    with pytest.raises(AppError) as caught, Database(path).connection():
        pass
    assert caught.value.code == 'MIGRATION_BUSY'
    with pytest.raises(AppError) as caught:
        migrations.initialize_database(path)
    assert caught.value.code == 'MIGRATION_BUSY'
    with pytest.raises(AppError):
        migrations.migrate_database(path, tmp_path/'backup.db')
    assert lock.read_text(encoding='utf-8') == 'maintenance'


def test_live_worker_blocks_migration_and_preserves_destination(tmp_path):
    from stockrl_app.models import utc_now
    path, backup = tmp_path/'old.db', tmp_path/'backup.db'
    old = old_database(path)
    old.execute('INSERT INTO worker_instances(owner_token,started_at,heartbeat_at,payload_json) VALUES (?,?,?,?)', ('owner','then',utc_now(),'{}'))
    with pytest.raises(AppError) as caught:
        migrations.migrate_database(path, backup)
    assert caught.value.code == 'MIGRATION_BUSY'
    assert not backup.exists()
    assert not path.with_name(path.name + '.migration.lock').exists()
    assert old.execute('PRAGMA user_version').fetchone()[0] == 1
    old.close()


def test_migration_preserves_incoming_references_and_research_jobs(tmp_path):
    path = tmp_path/'old.db'
    old = old_database(path)
    old.execute("INSERT INTO job_events VALUES ('j',1,'now','state','event bytes')")
    old.execute("INSERT INTO artifacts VALUES ('art','e','output','result.json','summary',12,'hash','artifact bytes')")
    old.execute("INSERT INTO idempotency_keys VALUES ('key','hash','e','j','now')")
    migrations.migrate_database(path, tmp_path/'backup.db')
    with Database(path).transaction() as c:
        assert c.execute('SELECT payload_json FROM job_events').fetchone()[0] == 'event bytes'
        assert c.execute('SELECT payload_json FROM artifacts').fetchone()[0] == 'artifact bytes'
        assert tuple(c.execute('SELECT experiment_id,job_id FROM idempotency_keys').fetchone()) == ('e','j')
        c.execute("INSERT INTO experiments(experiment_id,created_at,kind,integrity,payload_json) VALUES ('r','now','research','pending','{}')")
        for job in ('resume1','resume2'):
            c.execute("INSERT INTO jobs(job_id,experiment_id,kind,status,created_at,payload_json) VALUES (?,'r','research','queued','now','{}')", (job,))
        c.execute("UPDATE experiments SET job_id='resume2' WHERE experiment_id='r'")
        assert c.execute('PRAGMA foreign_key_check').fetchall() == []
    old.close()
