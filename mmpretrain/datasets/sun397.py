# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import SUN397_CATEGORIES

# Note that some images are not a jpg file although the name ends
# with jpg and therefore cannot be read properly. So we provide
# a list to skip these files.
INVALID = [
    '/a/assembly_line/sun_ajckcfldgdrdjogj.jpg',
    '/a/auto_factory/sun_apfsprenzdnzbhmt.jpg',
    '/b/baggage_claim/sun_avittiqqaiibgcau.jpg',
    '/b/batters_box/sun_alqlfpgtbgggezyr.jpg',
    '/b/bow_window/indoor/sun_ahsholsagvlrsboa.jpg',
    '/b/bow_window/indoor/sun_aioomcoujmmcxkkx.jpg',
    '/b/bow_window/outdoor/sun_atgtjdpqikjmllth.jpg',
    '/c/carrousel/sun_atsgphqympojgxnc.jpg',
    '/c/carrousel/sun_auzitjuirwolazns.jpg',
    '/c/church/outdoor/sun_boagasgfltequmal.jpg',
    '/c/church/outdoor/sun_brhmnwzzbkphcvfo.jpg',
    '/c/church/outdoor/sun_byjkqzybxpjnuofa.jpg',
    '/c/corridor/sun_aznefxvocwpgimko.jpg',
    '/d/dentists_office/sun_aaefsoauqlcsihou.jpg',
    '/d/diner/indoor/sun_apswilaujhntrybg.jpg',
    '/e/elevator/door/sun_aaudobqlphijkjdv.jpg',
    '/f/fastfood_restaurant/sun_axeniwtesffxqedr.jpg',
    '/f/fire_station/sun_bjyapttwilyyuxqm.jpg',
    '/f/fountain/sun_axgmpbdyvqhtkhee.jpg',
    '/h/hospital_room/sun_ahokhhxjiclpxqqa.jpg',
    '/o/oast_house/sun_bqsrrygxyrutgjve.jpg',
    '/r/restaurant_patio/sun_aurwypviprwycame.jpg',
    '/s/ski_resort/sun_bplmntyzoiobcqhp.jpg',
    '/w/wine_cellar/bottle_storage/sun_afmzwxkzmxkbamqi.jpg',
    '/w/wine_cellar/bottle_storage/sun_ahyymswdjejrbhyb.jpg',
    '/w/wine_cellar/bottle_storage/sun_avnttpxamufejbfe.jpg',
    '/a/archive/sun_awgsrbljlsvhqjij.jpg',
    '/a/art_school/sun_aabogqsjulyvmcse.jpg',
    '/a/art_school/sun_apnzojafyvkariue.jpg',
    '/b/ball_pit/sun_atjhwqngtoeuwhso.jpg',
    '/b/bow_window/indoor/sun_asxvsqbexmmtqmht.jpg',
    '/b/bow_window/indoor/sun_abeugxecxrwzmffp.jpg',
    '/b/bow_window/outdoor/sun_auwcqhrtzkgihvlv.jpg',
    '/b/bow_window/outdoor/sun_apnvdyecnjjmcuhi.jpg',
    '/c/childs_room/sun_alggivksjwwiklmt.jpg',
    '/c/control_tower/outdoor/sun_avbcxakrvpomqdgr.jpg',
    '/d/diner/indoor/sun_ajmzozstvsxisvgx.jpg',
    '/e/elevator/door/sun_aaqsyluqbluugqgy.jpg',
    '/f/fastfood_restaurant/sun_aevchxlxoruhxgrb.jpg',
    '/f/firing_range/indoor/sun_affrzvahwjorpalo.jpg',
    '/f/formal_garden/sun_bjvrlaeatjufekft.jpg',
    '/g/garage/indoor/sun_akbocuwclkxqlofx.jpg',
    '/g/greenhouse/indoor/sun_addirvgtxfbndlwf.jpg',
    '/k/kindergarden_classroom/sun_ajtpaahilrqzarri.jpg',
    '/l/laundromat/sun_afrrjykuhhlwiwun.jpg',
    '/m/music_studio/sun_bsntklkmwqgnjrjj.jpg',
    '/t/track/outdoor/sun_aophkoiosslinihb.jpg',
    '/a/archive/sun_aegmzltkiwyevpwa.jpg',
    '/a/auto_factory/sun_aybymzvbxgvcrwgn.jpg',
    '/b/baggage_claim/sun_atpmiqmnxjpgqsxi.jpg',
    '/b/baggage_claim/sun_ajffcdpsvgqfzoxx.jpg',
    '/b/bamboo_forest/sun_ausmxphosyahoyjo.jpg',
    '/b/batters_box/sun_aaeheulsicxtxnbu.jpg',
    '/c/carrousel/sun_arjrjcxemhttubqz.jpg',
    '/c/chicken_coop/outdoor/sun_abcegmmdbizqkpgh.jpg',
    '/c/control_tower/outdoor/sun_axhjfpkxdvqdfkyr.jpg',
    '/d/diner/indoor/sun_apaotiublwqeowck.jpg',
    '/f/fastfood_restaurant/sun_anexashcgmxdbmxq.jpg',
    '/l/landing_deck/sun_aizahnjfkuurjibw.jpg',
    '/n/nuclear_power_plant/outdoor/sun_aoblfvgyleweqanr.jpg',
    '/w/waiting_room/sun_aicytusmthfvqcwc.jpg',
    '/b/bow_window/indoor/sun_asmvdfnjlulewkpr.jpg',
    '/b/bus_interior/sun_adhktvidwzmodeou.jpg',
    '/c/catacomb/sun_algnawesgjzzmcqd.jpg',
    '/c/church/outdoor/sun_baihxlseimcsdhdx.jpg',
    '/d/diner/indoor/sun_agoyalzcawgxodbm.jpg',
    '/e/elevator_shaft/sun_awaitimkinrjaybl.jpg',
    '/f/fastfood_restaurant/sun_aplvzfbmtqtbsvbx.jpg',
    '/g/greenhouse/indoor/sun_bkccvyfpwetwjuhk.jpg',
    '/c/car_interior/backseat/sun_adexwfoqdyhowxpu.jpg',
    '/c/church/outdoor/sun_blmmweiumednscuf.jpg',
    '/f/fire_station/sun_bibntbsuunbsdrum.jpg',
    '/g/game_room/sun_aopfaqlllpvzhrak.jpg',
    '/u/underwater/coral_reef/sun_biiueajvszaxqopo.jpg',
    '/a/airplane_cabin/sun_arqyikigkyfpegug.jpg',
    '/b/badminton_court/indoor/sun_amppvxecgtjpfold.jpg',
    '/c/carrousel/sun_anxtrtieimkpmhvk.jpg',
    '/c/computer_room/sun_aebgvpgtwoqbfyvl.jpg',
    '/f/fire_escape/sun_atbraxuwwlvdoolv.jpg',
    '/k/kasbah/sun_abxkkoielpavsouu.jpg',
    '/t/tower/sun_bccqnzcvqkiwicjt.jpg',
    '/a/archive/sun_afngadshxudodkct.jpg',
    '/b/bow_window/indoor/sun_awnrlipyxpgxxgxz.jpg',
    '/c/control_tower/outdoor/sun_arohngcbtsvbthho.jpg',
    '/f/fire_station/sun_brbskkfgghbfvgkk.jpg',
    '/r/restaurant_patio/sun_amjfbqzfgxarrpec.jpg',
    '/v/vineyard/sun_bdxhnbgbnolddswz.jpg',
    '/b/baggage_claim/sun_axrtsmillrglugia.jpg',
    '/d/diner/indoor/sun_alaqevbwpjaqqdqz.jpg',
    '/l/landing_deck/sun_acodgoamhgnnbmvr.jpg',
    '/c/carrousel/sun_adsafgyrinnekycc.jpg',
    '/c/church/outdoor/sun_bzqhuwshtdgakkay.jpg',
    '/c/closet/sun_absahzamlrylkxyn.jpg',
    '/f/fire_escape/sun_acdthenaosuqcoqn.jpg',
    '/b/butchers_shop/sun_asrdgbefoszenfex.jpg',
    '/c/church/outdoor/sun_bzfyucfrdigaqneg.jpg',
    '/c/church/outdoor/sun_byzxhknqrejdajxi.jpg',
    '/c/cockpit/sun_ajkulpqauavrmxae.jpg',
    '/l/living_room/sun_aefoqbeatyufobtx.jpg',
    '/s/supermarket/sun_attvxbzocurnddbz.jpg',
    '/c/closet/sun_aqnutmwfkypmrnfy.jpg',
    '/f/fire_station/sun_bttrtzktpbymxkmf.jpg',
    '/s/shopping_mall/indoor/sun_avwzjsijaxnwuzjx.jpg',
    '/w/windmill/sun_blvczkyqbmabzeej.jpg',
    '/c/chicken_coop/outdoor/sun_amaonsnnkskxwmrj.jpg',
    '/s/swimming_pool/outdoor/sun_bslaihiqlhfewtzn.jpg',
    '/u/underwater/coral_reef/sun_bhcrnmvbgnkvcvkr.jpg',
    '/d/dining_room/sun_azlxdhiajwrhaivq.jpg',
    '/c/church/outdoor/sun_bnunxbznqnvgeykx.jpg',
    '/c/corridor/sun_aspwpqqlcwzfanvl.jpg',
    '/r/restaurant_patio/sun_awcbpizjbudjvrhs.jpg',
    '/b/ball_pit/sun_avdnmemjrgrbkwjm.jpg',
]


@DATASETS.register_module()
class SUN397(BaseDataset):
    """The SUN397 Dataset.

    Support the `SUN397 Dataset <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    SUN397 dataset directory: ::

        SUN397
        ├── SUN397
        │   ├── a
        │   │   ├── abbey
        │   |   |   ├── sun_aaalbzqrimafwbiv.jpg
        │   |   |   └── ...
        │   │   ├── airplane_cabin
        │   |   |   ├── sun_aadqdkqaslqqoblu.jpg
        │   |   |   └── ...
        │   |   └── ...
        │   ├── b
        │   │   └── ...
        │   ├── c
        │   │   └── ...
        │   └── ...
        └── Partitions
            ├── ClassName.txt
            ├── Training_01.txt
            ├── Testing_01.txt
            └── ...

    Args:
        data_root (str): The root directory for Stanford Cars dataset.
        split (str, optional): The dataset split, supports "train" and "test".
            Default to "train".

    Examples:
        >>> from mmpretrain.datasets import SUN397
        >>> train_dataset = SUN397(data_root='data/SUN397', split='train')
        >>> train_dataset
        Dataset SUN397
            Number of samples:  19824
            Number of categories:       397
            Root of dataset:    data/SUN397
        >>> test_dataset = SUN397(data_root='data/SUN397', split='test')
        >>> test_dataset
        Dataset SUN397
            Number of samples:  19829
            Number of categories:       397
            Root of dataset:    data/SUN397
    """  # noqa: E501

    METAINFO = {'classes': SUN397_CATEGORIES}

    def __init__(self, data_root: str, split: str = 'train', **kwargs):

        splits = ['train', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        self.backend = get_file_backend(data_root, enable_singleton=True)
        if split == 'train':
            ann_file = self.backend.join_path('Partitions', 'Training_01.txt')
        else:
            ann_file = self.backend.join_path('Partitions', 'Testing_01.txt')

        data_prefix = 'SUN397'
        test_mode = split == 'test'

        super(SUN397, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self):
        pairs = list_from_file(self.ann_file)
        data_list = []
        for pair in pairs:
            if pair in INVALID:
                continue
            img_path = self.backend.join_path(self.img_prefix, pair[1:])
            items = pair.split('/')
            class_name = '_'.join(items[2:-1])
            gt_label = self.METAINFO['classes'].index(class_name)
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
