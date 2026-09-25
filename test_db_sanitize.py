"""Tests for db._sanitize_nan: NaN/Inf -> None and NUL stripping before Supabase writes.

No network: requests.post is mocked.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from db import SupabaseClient, _sanitize_nan


NUL = '\x00'


# --- _sanitize_nan: NUL stripping -------------------------------------------

def test_strips_nul_from_top_level_string():
    assert _sanitize_nan(f'ab{NUL}c{NUL}') == 'abc'


def test_strips_nul_from_nested_dict_value():
    data = {'outer': {'inner': f'Acme{NUL} Ltd'}}
    assert _sanitize_nan(data) == {'outer': {'inner': 'Acme Ltd'}}


def test_strips_nul_from_string_inside_list():
    data = {'skills': ['python', f'go{NUL}lang', [f'{NUL}rust']]}
    assert _sanitize_nan(data) == {'skills': ['python', 'golang', ['rust']]}


def test_strips_nul_from_dict_keys_including_nested():
    data = {f'na{NUL}me': 'x', 'raw': {f'{NUL}title': {f'deep{NUL}': 1}}}
    assert _sanitize_nan(data) == {'name': 'x', 'raw': {'title': {'deep': 1}}}


def test_colliding_keys_after_strip_keep_last_value():
    data = {f'a{NUL}': 1, 'a': 2}
    assert _sanitize_nan(data) == {'a': 2}


def test_tuple_becomes_list_and_is_cleaned():
    assert _sanitize_nan((f'x{NUL}', float('nan'), 1)) == ['x', None, 1]


# --- _sanitize_nan: NaN / Inf unchanged -------------------------------------

def test_nan_and_inf_become_none():
    data = {
        'a': float('nan'),
        'b': float('inf'),
        'c': float('-inf'),
        'lst': [1, float('nan'), float('inf'), float('-inf'), 3],
        'nested': {'d': float('nan')},
    }
    assert _sanitize_nan(data) == {
        'a': None, 'b': None, 'c': None,
        'lst': [1, None, None, None, 3],
        'nested': {'d': None},
    }


# --- _sanitize_nan: everything else untouched -------------------------------

def test_non_string_values_untouched():
    data = {'i': 42, 't': True, 'f': False, 'n': None, 'x': 3.5, 0: 'int key'}
    out = _sanitize_nan(data)
    assert out == data
    assert out['t'] is True and out['f'] is False and out['n'] is None
    assert isinstance(out['i'], int) and not isinstance(out['i'], bool)


def test_clean_input_comes_out_equal():
    data = {
        'name': 'Dana Cohen',
        'titles': ['Engineer', 'Lead'],
        'raw': {'score': 7.5, 'years': 4, 'ok': True, 'none': None},
    }
    assert _sanitize_nan(data) == data


def test_json_dumps_output_has_no_u0000():
    data = {f'k{NUL}': [f'v{NUL}', {'n': f'{NUL}{NUL}'}], 'nan': float('nan')}
    out = json.dumps(_sanitize_nan(data), allow_nan=False)
    assert '\\u0000' not in out
    assert json.loads(out) == {'k': ['v', {'n': ''}], 'nan': None}


# --- SupabaseClient write paths ---------------------------------------------

def _dirty_profile():
    return {
        'linkedin_url': 'https://www.linkedin.com/in/test-person',
        'name': f'Test{NUL} Person',
        'raw_data': {
            f'head{NUL}line': f'Engineer{NUL}',
            'skills': [f'py{NUL}thon', 'go'],
            'score': float('nan'),
        },
    }


def _mock_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = '[]'
    resp.json.return_value = []
    return resp


def _assert_clean_body(mock_post):
    assert mock_post.call_count == 1
    body = mock_post.call_args.kwargs['data']
    assert '\\u0000' not in body
    assert NUL not in body
    parsed = json.loads(body)  # must parse cleanly
    return parsed


@pytest.fixture
def client():
    return SupabaseClient('https://example.supabase.co', 'test-key')


def test_rpc_body_has_no_nul(client):
    with patch('db.requests.post', return_value=_mock_response()) as mock_post:
        client.rpc('sourcingx_upsert_profiles', {'profiles': [_dirty_profile()]})
    parsed = _assert_clean_body(mock_post)
    raw = parsed['profiles'][0]['raw_data']
    assert raw['headline'] == 'Engineer'
    assert raw['skills'] == ['python', 'go']
    assert raw['score'] is None


def test_upsert_body_has_no_nul(client):
    with patch('db.requests.post', return_value=_mock_response()) as mock_post:
        client.upsert('profiles', _dirty_profile(), on_conflict='linkedin_url')
    parsed = _assert_clean_body(mock_post)
    assert parsed['name'] == 'Test Person'
    assert parsed['raw_data']['headline'] == 'Engineer'


def test_upsert_batch_body_has_no_nul(client):
    with patch('db.requests.post', return_value=_mock_response()) as mock_post:
        client.upsert_batch('profiles', [_dirty_profile(), _dirty_profile()],
                            on_conflict='linkedin_url')
    parsed = _assert_clean_body(mock_post)
    assert len(parsed) == 2
    for row in parsed:
        assert row['name'] == 'Test Person'
        assert 'headline' in row['raw_data']
        assert row['raw_data']['score'] is None


# --- SupabaseClient._request (insert/update) --------------------------------

def _mock_request_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = '[]'
    resp.json.return_value = []
    resp.raise_for_status.return_value = None
    return resp


def _assert_clean_json_kwarg(mock_request):
    assert mock_request.call_count == 1
    payload = mock_request.call_args.kwargs['json']
    dumped = json.dumps(payload)
    assert '\\u0000' not in dumped

    def _no_nul(value):
        if isinstance(value, str):
            assert NUL not in value
        elif isinstance(value, dict):
            for k, v in value.items():
                assert NUL not in k
                _no_nul(v)
        elif isinstance(value, list):
            for v in value:
                _no_nul(v)

    _no_nul(payload)
    return payload


def test_insert_json_kwarg_has_no_nul(client):
    with patch('db.requests.request', return_value=_mock_request_response()) as mock_request:
        client.insert('profiles', _dirty_profile())
    payload = _assert_clean_json_kwarg(mock_request)
    assert payload['name'] == 'Test Person'
    assert payload['raw_data']['headline'] == 'Engineer'
    assert payload['raw_data']['score'] is None


def test_update_json_kwarg_has_no_nul(client):
    with patch('db.requests.request', return_value=_mock_request_response()) as mock_request:
        client.update('profiles', _dirty_profile(), {'linkedin_url': 'https://www.linkedin.com/in/test-person'})
    payload = _assert_clean_json_kwarg(mock_request)
    assert payload['name'] == 'Test Person'
    assert payload['raw_data']['headline'] == 'Engineer'
    assert payload['raw_data']['score'] is None
