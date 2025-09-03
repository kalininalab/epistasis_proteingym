from pathlib import Path
import pandas as pd
import re

def convert_name_to_gfp(name):
    return {'D7PM05_CLYGR_Somermeyer_2022': 'cgreGFP',
            'Q6WV12_9MAXI_Somermeyer_2022': 'ppluGFP',
            'Q8WTC7_9CNID_Somermeyer_2022': 'amacGFP'}.get(name, name)
    
    
def convert_name_tsuboyama(csv_path: Path) -> str:
    parts = Path(csv_path).stem.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[-1]}"
    return Path(csv_path).stem


def shift_mutation_positions_up(mutant_str):
    if pd.isna(mutant_str):
        return mutant_str
    mutations = mutant_str.split(':')
    shifted = []
    for m in mutations:
        match = re.match(r"([A-Z])(\d+)([A-Z])", m)
        if match:
            aa_from, pos, aa_to = match.groups()
            shifted_pos = str(int(pos) + 1)
            shifted.append(f"{aa_from}{shifted_pos}{aa_to}")
        else:
            shifted.append(m)
    return ':'.join(shifted)