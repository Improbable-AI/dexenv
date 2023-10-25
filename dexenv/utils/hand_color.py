import numpy as np

FINGERTIP_COLORS = np.array([
    [111, 29, 27],
    [187, 148, 87],
    [67, 40, 24],
    [153, 88, 42],
    [255, 230, 167]
]) / 255.0
FINGERTIP_COLORS = FINGERTIP_COLORS.tolist()

shadowhand_body_ids = {
    'robot0:ffdistal': 7,
    'robot0:ffknuckle': 4,
    'robot0:ffmiddle': 6,
    'robot0:ffproximal': 5,
    'robot0:forearm': 1,
    'robot0:hand mount': 0,
    'robot0:lfdistal': 20,
    'robot0:lfknuckle': 17,
    'robot0:lfmetacarpal': 16,
    'robot0:lfmiddle': 19,
    'robot0:lfproximal': 18,
    'robot0:mfdistal': 11,
    'robot0:mfknuckle': 8,
    'robot0:mfmiddle': 10,
    'robot0:mfproximal': 9,
    'robot0:palm': 3,
    'robot0:rfdistal': 15,
    'robot0:rfknuckle': 12,
    'robot0:rfmiddle': 14,
    'robot0:rfproximal': 13,
    'robot0:thbase': 21,
    'robot0:thdistal': 25,
    'robot0:thhub': 23,
    'robot0:thmiddle': 24,
    'robot0:thproximal': 22,
    'robot0:wrist': 2
}

shadowhand_body_color_mapping = {
    'robot0:ffdistal': [111, 29, 27],
    'robot0:ffknuckle': [255, 190, 11],
    'robot0:ffmiddle': [155, 93, 229],
    'robot0:ffproximal': [252, 170, 103],
    'robot0:forearm': [43, 48, 53],
    'robot0:hand mount': [0, 0, 0],
    'robot0:lfdistal': [187, 148, 87],
    'robot0:lfknuckle': [251, 86, 7],
    'robot0:lfmetacarpal': [204, 255, 51],
    'robot0:lfmiddle': [241, 91, 181],
    'robot0:lfproximal': [176, 65, 62],
    'robot0:mfdistal': [67, 40, 24],
    'robot0:mfknuckle': [255, 0, 110],
    'robot0:mfmiddle': [154, 180, 64],  # [254, 228, 64],
    'robot0:mfproximal': [255, 255, 199],
    # 'robot0:palm': [108, 117, 125],
    'robot0:palm': [0, 75, 35],
    'robot0:rfdistal': [153, 88, 42],
    'robot0:rfknuckle': [131, 56, 236],
    'robot0:rfmiddle': [0, 138, 184],
    'robot0:rfproximal': [84, 134, 135],
    'robot0:thbase': [0, 114, 0],
    'robot0:thdistal': [255, 230, 167],
    'robot0:thhub': [58, 134, 255],
    'robot0:thmiddle': [154, 230, 234],
    'robot0:thproximal': [106, 77, 80],
    'robot0:wrist': [255, 213, 0]
}

dclaw_body_color_mapping = {
    'two1_link': [255, 0, 110],
    'two2_link': [255, 190, 11],
    'two3_link': [155, 93, 229],
    'two0_link': [0, 75, 35],
    'three1_link': [252, 170, 103],
    'three2_link': [176, 65, 62],
    'three3_link': [84, 134, 135],
    'three0_link': [0, 138, 184],
    'one3_link': [58, 134, 255],
    'one2_link': [154, 230, 234],
    'one1_link': [106, 77, 80],
    'one0_link': [255, 255, 199],
    'four3_link': [58, 134, 255],
    'four2_link': [154, 230, 234],
    'four1_link': [106, 77, 80],
    'four0_link': [255, 255, 199],
    'five1_link': [252, 170, 103],
    'five2_link': [176, 65, 62],
    'five3_link': [84, 134, 135],
    'five0_link': [0, 138, 184],
    'base_link': [43, 48, 53]
}

for k, v in shadowhand_body_color_mapping.items():
    v = np.array(v) / 255.0
    shadowhand_body_color_mapping[k] = v.tolist()

for k, v in dclaw_body_color_mapping.items():
    v = np.array(v) / 255.0
    dclaw_body_color_mapping[k] = v.tolist()
