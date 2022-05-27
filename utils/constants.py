__all__ = ['COCO_CAT_NAMES', 'CAT_ID_TO_NAME', 'CAT_NAME_TO_ID', 'CAT_ID_TO_IDX', 'EVAL_SPLITS_DICT']


COCO_CAT_NAMES = [
    '__background__',
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 
    'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 
    'swimming-pool', 'helicopter', 'water-tower']
assert len(COCO_CAT_NAMES) == 17

CAT_ID_TO_NAME = {
    0: 'plane', 1: 'baseball-diamond', 2: 'bridge', 3: 'ground-track-field', 4: 'small-vehicle', 5: 'large-vehicle', 
    6: 'ship', 7: 'tennis-court', 8: 'basketball-court', 9: 'storage-tank', 10: 'soccer-ball-field', 11: 'roundabout', 12: 'harbor', 
    13: 'swimming-pool', 14: 'helicopter', 15: 'water-tower'}

assert len(CAT_ID_TO_NAME) == 16

CAT_NAME_TO_ID = {}
for k, v in CAT_ID_TO_NAME.items():
    CAT_NAME_TO_ID[v] = k
    assert v in COCO_CAT_NAMES
assert len(CAT_NAME_TO_ID) == 16

CAT_ID_TO_IDX = {}
for k, v in CAT_ID_TO_NAME.items():
    CAT_ID_TO_IDX[k] = COCO_CAT_NAMES.index(v)
assert len(CAT_ID_TO_IDX) == 16

EVAL_SPLITS_DICT = {
    'rsvg': ['val', 'test']
}
