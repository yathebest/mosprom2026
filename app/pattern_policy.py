"""Production capabilities, independent of RunConfig and historical records.

The manifest is an explicit allowlist: omitted patterns cannot be activated.
Missing or malformed policy cannot restore previously disabled scenarios.
"""
from copy import deepcopy
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name('pattern-policy.json')

_SPECS = (
    ('helmet', 'helmet_detection', ('no_helmet',), ()),
    ('vest', 'vest_detection', ('no_vest',), ()),
    ('mask', 'mask_detection', ('no_mask',), ()),
    ('harness', 'harness_detection', ('no_harness_at_height',), ('height',)),
    ('phone', 'phone_detection', ('possible_phone_use',), ()),
    ('fire', 'fire_detection', ('fire',), ()),
    ('smoke', 'smoke_detection', ('smoke',), ()),
    ('smoking', 'smoking_detection', ('possible_smoking',), ()),
    ('proximity', 'proximity_detection', ('equipment_proximity',), ()),
    ('load_posture', 'load_posture_detection', ('unsafe_load_posture',), ()),
    ('fall', 'fall_detection', ('possible_fall', 'person_down'), ()),
    ('fast_motion', 'fast_motion', ('fast_motion',), ()),
    ('danger_zone', None, ('danger_zone',), ('danger_zone',)),
    ('barrier', None, ('barrier_crossing',), ('barrier',)),
    ('under_load', None, ('under_load',), ('under_load',)),
    ('under_conveyor', None, ('under_conveyor',), ('under_conveyor',)),
)
_OPTIONS = {'phone_work_context_only': {
    'pattern_id': 'phone_context',
    'reason': 'Ограничение поиска телефона движением или рабочим контекстом отключено',
}}


class PolicyError(ValueError):
    pass


def catalog():
    """Read a single policy snapshot; malformed policy never enables a bypass."""
    known = {row[0] for row in _SPECS}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
            version = manifest['version']
            selected = manifest['enabled_patterns']
            decisions = manifest.get('decisions', {})
            disabled_options = manifest.get('disabled_options', [])
            if (not isinstance(version, str) or not version.strip()
                    or not isinstance(selected, list) or any(not isinstance(k, str) for k in selected)
                    or set(selected)-known or len(set(selected)) != len(selected)
                    or not isinstance(decisions, dict) or set(decisions)-known
                    or any(not isinstance(v, dict) for v in decisions.values())
                    or not isinstance(disabled_options, list)
                    or any(not isinstance(key, str) for key in disabled_options)
                    or set(disabled_options)-set(_OPTIONS)
                    or len(set(disabled_options)) != len(disabled_options)):
                raise ValueError('Invalid pattern policy schema')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PolicyError('Не удалось прочитать утверждённый перечень сценариев') from exc
        enabled = set(selected)
    else:
        raise PolicyError('Отсутствует утверждённый перечень сценариев')
    patterns = []
    for key, flag, kinds, zones in _SPECS:
        decision = decisions.get(key, {})
        patterns.append({'id': key, 'enabled': key in enabled, 'config_flag': flag,
            'event_kinds': list(kinds), 'zone_kinds': list(zones),
            'reason': str(decision.get('reason', '' if key in enabled else 'Сценарий не включён в утверждённый перечень')),
            'evidence_status': str(decision.get('evidence_status', 'not_measured'))})
    zones = {zone for row in patterns if row['enabled'] for zone in row['zone_kinds']}
    zones.add('reflection')  # Technical detector mask, not an alert pattern.
    if 'proximity' in enabled or ('phone' in enabled and 'phone_work_context_only' not in disabled_options):
        zones.add('equipment')
    if 'smoking' in enabled:
        zones.add('smoking_allowed')
    if enabled & {'fire', 'smoke'}:
        zones.add('fire_allowed')
    return {'version': version, 'patterns': patterns,
        'enabled_flags': [row['config_flag'] for row in patterns if row['enabled'] and row['config_flag']],
        'disabled_flags': [row['config_flag'] for row in patterns if not row['enabled'] and row['config_flag']]+disabled_options,
        'disabled_options': list(disabled_options),
        'enabled_zone_kinds': sorted(zones)}


def disabled_settings(config, policy=None):
    policy = policy or catalog()
    data = config.model_dump() if hasattr(config, 'model_dump') else config
    warnings = []
    for row in policy['patterns']:
        if not row['enabled'] and row['config_flag'] and data.get(row['config_flag']):
            warnings.append({'pattern_id': row['id'], 'setting': row['config_flag'], 'reason': row['reason']})
    for option in policy.get('disabled_options', []):
        if data.get(option):
            warnings.append({'pattern_id': _OPTIONS[option]['pattern_id'], 'setting': option,
                             'reason': _OPTIONS[option]['reason']})
    owners = {zone: row for row in policy['patterns'] for zone in row['zone_kinds']}
    for zone in data.get('zones', []):
        kind = zone.get('kind', 'danger_zone')
        if kind not in policy['enabled_zone_kinds']:
            row = owners.get(kind)
            warnings.append({'pattern_id': row['id'] if row else kind, 'setting': 'zones',
                'zone_kind': kind, 'zone_name': zone.get('name', ''),
                'reason': row['reason'] if row else 'Контекст отключённого сценария'})
    return warnings


def sanitize_config(config, policy=None):
    """Return an editable copy and explicit warnings; never rewrite stored data."""
    policy = policy or catalog()
    data = deepcopy(config.model_dump() if hasattr(config, 'model_dump') else config)
    warnings = disabled_settings(data, policy)
    for flag in policy['disabled_flags']:
        if flag in data:
            data[flag] = False
    if 'zones' in data:
        data['zones'] = [zone for zone in data['zones'] if zone.get('kind', 'danger_zone') in policy['enabled_zone_kinds']]
    if 'state_windows' in data:
        allowed = {kind for row in policy['patterns'] if row['enabled'] for kind in row['event_kinds']}
        data['state_windows'] = {key: value for key, value in data['state_windows'].items() if key in allowed}
    return data, warnings
