from pathlib import Path

def convert_name_to_gfp(name):
    return {'D7PM05_CLYGR_Somermeyer_2022': 'cgreGFP',
            'Q6WV12_9MAXI_Somermeyer_2022': 'ppluGFP',
            'Q8WTC7_9CNID_Somermeyer_2022': 'amacGFP'}.get(name, name)
    
    
def convert_name_tsuboyama(csv_path: Path) -> str:
    parts = Path(csv_path).stem.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[-1]}"
    return Path(csv_path).stem