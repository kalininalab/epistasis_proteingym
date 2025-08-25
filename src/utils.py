def convert_name_to_gfp(name):
    return {'D7PM05_CLYGR_Somermeyer_2022': 'cgreGFP',
            'Q6WV12_9MAXI_Somermeyer_2022': 'ppluGFP',
            'Q8WTC7_9CNID_Somermeyer_2022': 'amacGFP'}.get(name, name)