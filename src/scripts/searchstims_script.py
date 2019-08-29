from pathlib import Path

import searchstims.make
from searchstims.stim_makers import RVvGVStimMaker, RVvRHGVStimMaker, Two_v_Five_StimMaker

IKKN_SIZE = (70, 70)
BORDER_SIZE = (5, 5)
GRID_SIZE = (2, 2)
ITEM_BBOX_SIZE = (30, 30)
JITTER = 12

for window_size in [IKKN_SIZE,]:
    keys = ['RVvGV', 'RVvRHGV', '2_v_5']
    vals = [
        RVvGVStimMaker(target_color='red',
                       distractor_color='green',
                       window_size=window_size,
                       border_size=BORDER_SIZE,
                       grid_size=GRID_SIZE,
                       item_bbox_size=ITEM_BBOX_SIZE,
                       jitter=JITTER),
        RVvRHGVStimMaker(target_color='red',
                         distractor_color='green',
                         window_size=window_size,
                         border_size=BORDER_SIZE,
                         grid_size=GRID_SIZE,
                         item_bbox_size=ITEM_BBOX_SIZE,
                         jitter=JITTER),
        Two_v_Five_StimMaker(target_color='white',
                             distractor_color='white',
                             window_size=window_size,
                             border_size=BORDER_SIZE,
                             grid_size=GRID_SIZE,
                             item_bbox_size=ITEM_BBOX_SIZE,
                             jitter=JITTER,
                             target_number=2,
                             distractor_number=5)
    ]
    if window_size == IKKN_SIZE:
        ikkn_zip = zip(keys, vals)

OUTPUT_DIR = Path('data/visual_search_stimuli')
TARGET_PRESENT = 20
TARGET_ABSENT = 20
SET_SIZES = [1, 2, 4]


def main():
    for net, zipped in zip(['ikkn', ], [ikkn_zip, ]):
        for key, val in zipped:
            json_filename = f'{net}_{key}.json'
            stim_dict = {key: val}
            output_dir = OUTPUT_DIR.joinpath(f'{net}_{key}')
            searchstims.make.make(root_output_dir=output_dir,
                                  stim_dict=stim_dict,
                                  json_filename=json_filename,
                                  num_target_present=TARGET_PRESENT,
                                  num_target_absent=TARGET_ABSENT,
                                  set_sizes=SET_SIZES)


if __name__ == '__main__':
    main()
