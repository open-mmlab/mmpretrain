# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class CUB(BaseDataset):
    """The CUB-200-2011 Dataset.

    Support the `CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
    Comparing with the `CUB-200 <http://www.vision.caltech.edu/visipedia/CUB-200.html>`_ Dataset,
    there are much more pictures in `CUB-200-2011`.

    Args:
        ann_file (str): the annotation file.
            images.txt in CUB.
        image_class_labels_file (str): the label file.
            image_class_labels.txt in CUB.
        train_test_split_file (str): the split file.
            train_test_split_file.txt in CUB.
    """  # noqa: E501

    CLASSES = [
        'Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross',
        'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet',
        'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird',
        'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird',
        'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting',
        'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat',
        'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant',
        'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird',
        'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow',
        'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo',
        'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker',
        'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher',
        'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher',
        'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird',
        'Northern_Fulmar', 'Gadwall', 'American_Goldfinch',
        'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe',
        'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak',
        'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak',
        'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull',
        'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull',
        'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird',
        'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear',
        'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay',
        'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird',
        'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher',
        'Ringed_Kingfisher', 'White_breasted_Kingfisher',
        'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard',
        'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser',
        'Mockingbird', 'Nighthawk', 'Clark_Nutcracker',
        'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole',
        'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican',
        'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit',
        'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven',
        'White_necked_Raven', 'American_Redstart', 'Geococcyx',
        'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow',
        'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow',
        'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow',
        'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow',
        'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow',
        'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow',
        'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow',
        'White_crowned_Sparrow', 'White_throated_Sparrow',
        'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow',
        'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager',
        'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern',
        'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee',
        'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo',
        'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo',
        'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo',
        'Bay_breasted_Warbler', 'Black_and_white_Warbler',
        'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler',
        'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler',
        'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler',
        'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler',
        'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler',
        'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler',
        'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler',
        'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush',
        'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing',
        'American_Three_toed_Woodpecker', 'Pileated_Woodpecker',
        'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker',
        'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren',
        'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren',
        'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat'
    ]

    def __init__(self, *args, ann_file, image_class_labels_file,
                 train_test_split_file, **kwargs):
        self.image_class_labels_file = image_class_labels_file
        self.train_test_split_file = train_test_split_file
        super(CUB, self).__init__(*args, ann_file=ann_file, **kwargs)

    def load_annotations(self):
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ')[1] for x in f.readlines()]

        with open(self.image_class_labels_file) as f:
            gt_labels = [
                # in the official CUB-200-2011 dataset, labels in
                # image_class_labels_file are started from 1, so
                # here we need to '- 1' to let them start from 0.
                int(x.strip().split(' ')[1]) - 1 for x in f.readlines()
            ]

        with open(self.train_test_split_file) as f:
            splits = [int(x.strip().split(' ')[1]) for x in f.readlines()]

        assert len(samples) == len(gt_labels) == len(splits),\
            f'samples({len(samples)}), gt_labels({len(gt_labels)}) and ' \
            f'splits({len(splits)}) should have same length.'

        data_infos = []
        for filename, gt_label, split in zip(samples, gt_labels, splits):
            if split and self.test_mode:
                # skip train samples when test_mode=True
                continue
            elif not split and not self.test_mode:
                # skip test samples when test_mode=False
                continue
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
