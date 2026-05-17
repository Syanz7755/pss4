import argparse
import re
from typing import List, Dict, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="1D-Periodic Planar FED Solver")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--all-freq", action="store_true", default=False)
    parser.add_argument("--out-dir", type=str, default="results/")
    parser.add_argument("input_path", type=str)
    return parser.parse_args()

def parse_stack_file(filepath: str) -> Tuple[List[Dict], List[Dict]]:
    layers = []
    groups = []
    in_periodic_boundary = False
    past_last_boundary = False
    layer_id = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('# group:'):
                if not past_last_boundary:
                    continue
                group_def = _parse_group_line(line, layers)
                if group_def is not None:
                    groups.append(group_def)
                continue
            
            if line.startswith('#'):
                continue
            
            if '#' in line:
                line = line.split('#')[0].strip()
            
            if line == 'PERIODIC_BOUNDARY':
                if in_periodic_boundary:
                    past_last_boundary = True
                in_periodic_boundary = not in_periodic_boundary
                continue
            
            if in_periodic_boundary and line:
                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                else:
                    parts = line.split()
                
                if len(parts) < 4:
                    continue
                
                material = parts[1] if parts[0].isdigit() else parts[0]
                thickness_idx = 2 if parts[0].isdigit() else 1
                is_source_idx = thickness_idx + 1
                is_probe_idx = thickness_idx + 2
                
                thickness_um = float(parts[thickness_idx])
                thickness = thickness_um * 1e-6
                is_source = parts[is_source_idx].lower() == 'true'
                is_probe = parts[is_probe_idx].lower() == 'true'
                
                if is_probe and material.lower() != 'vacuum':
                    raise ValueError("Probes can only be placed in vacuum layers.")
                
                layer = {
                    'layer_id': layer_id,
                    'material': material,
                    'thickness': thickness,
                    'is_source': is_source,
                    'is_probe': is_probe
                }
                layers.append(layer)
                layer_id += 1
    
    _validate_groups(groups, layers, filepath)
    return layers, groups

def _parse_group_line(line: str, layers: List[Dict]) -> Dict:
    content = line[len('# group:'):].strip()
    kv_pairs = {}
    for part in content.split(','):
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            kv_pairs[key.strip()] = value.strip()
    
    if 'name' not in kv_pairs or 'ids' not in kv_pairs:
        return None
    
    name = kv_pairs['name']
    ids_str = kv_pairs['ids']
    ids_match = re.match(r'\(([^)]+)\)', ids_str)
    if not ids_match:
        raise ValueError(f"Invalid ids format: {line}")
    ids = [int(x.strip()) for x in ids_match.group(1).split(',')]
    return {'name': name, 'ids': ids}

def _validate_groups(groups: List[Dict], layers: List[Dict], filepath: str):
    vacuum_layer_ids = set(layer['layer_id'] for layer in layers if layer['material'].lower() == 'vacuum')
    all_layer_ids = set(layer['layer_id'] for layer in layers)
    
    for group in groups:
        for sid in group['ids']:
            if sid in vacuum_layer_ids:
                import traceback
                traceback.print_stack()
                raise ValueError(
                    f"Error in '{filepath}': Group '{group['name']}' contains vacuum layer ID {sid}. "
                    f"Vacuum IDs {vacuum_layer_ids} cannot be in source groups. Group ids={group['ids']}")
            if sid not in all_layer_ids:
                import traceback
                traceback.print_stack()
                raise ValueError(
                    f"Error in '{filepath}': Group '{group['name']}' has invalid layer ID {sid}. "
                    f"Valid IDs: {sorted(all_layer_ids)}. Group ids={group['ids']}")
