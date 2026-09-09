import json

import pytest

from stockrl_app.errors import AppError
from stockrl_app.research_artifacts import seal_research_bundle, verify_research_bundle


def test_sealed_unit_detects_mutation_and_unlisted_files(tmp_path):
    (tmp_path / 'model.zip').write_bytes(b'fixture')
    digest = seal_research_bundle(tmp_path, {'research_id': 'r', 'protocol_sha256': 'a' * 64})
    assert verify_research_bundle(tmp_path, expected_hash=digest)['research_id'] == 'r'
    (tmp_path / 'extra.txt').write_text('not indexed')
    with pytest.raises(AppError, match='文件'):
        verify_research_bundle(tmp_path, expected_hash=digest)
    (tmp_path / 'extra.txt').unlink()
    (tmp_path / 'model.zip').write_bytes(b'changed')
    with pytest.raises(AppError):
        verify_research_bundle(tmp_path, expected_hash=digest)


def test_version_conflict_is_not_read_as_v1(tmp_path):
    (tmp_path / 'model.zip').write_bytes(b'fixture')
    seal_research_bundle(tmp_path, {})
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['versions']['artifact_schema_version'] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(AppError) as error:
        verify_research_bundle(tmp_path)
    assert error.value.code == 'UNSUPPORTED_ARTIFACT_VERSION'


def test_reserved_metadata_cannot_override_manifest(tmp_path):
    with pytest.raises(AppError):
        seal_research_bundle(tmp_path, {'schema_version': 1})
