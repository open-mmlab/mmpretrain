# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import re
from collections import defaultdict

import mmengine.registry.root as mmengine_registry_module
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.registry import Registry
from mmengine.utils import mkdir_or_exist
from rich.progress import track
from yapf.yapflib.yapf_api import FormatCode

import mmpretrain.registry as mmpretrain_registry_module
from mmpretrain.utils import register_all_modules as mmpretrain_register_all_modules

logger = MMLogger('convert', log_file='work_dirs/debug.log')
logger.handlers[1].setLevel('DEBUG')

SCOPE = 'mmpretrain'
TARGET_DIR = f'{SCOPE}/configs'
CONFIG_DIR = './configs'

SKIP_NAMES = set(['BN', 'DCN', 'DCNv2'])


def _parse_type_name(type_name):
    matched = re.match(r'(.*)\.(.*)', type_name)
    if matched is not None:
        scope, type_name = matched.groups()
    else:
        scope = None
    return scope, type_name


def _get_module(type_name, all_modules, scope=None):
    if scope is None:
        scope = SCOPE
    if type_name in all_modules[scope]:
        return all_modules[scope][type_name]
    else:
        return all_modules['mmengine'].get(type_name)


def _should_skip(register_name, register_module):
    if register_module is None:
        return True
    if register_name != register_module.__name__:
        return True
    elif 'mmcv.transforms' in register_module.__module__:
        return False
    elif 'mmcv' in register_module.__module__:
        return True
    else:
        return False


def collect_all_modules():
    mmpretrain_registries = [
        value for value in mmpretrain_registry_module.__dict__.values()
        if isinstance(value, Registry)
    ]
    mmengine_registries = [
        value for value in mmengine_registry_module.__dict__.values()
        if isinstance(value, Registry)
    ]
    all_registries = set(mmpretrain_registries + mmengine_registries)
    all_modules = defaultdict(dict)
    for registry in all_registries:
        all_modules[registry.scope].update(registry.module_dict)
    return all_modules


def replace_type_assign(file_content: str, all_modules):
    type_assigns = r"type='?(.*?)'?[\n\s,\)]"
    imported_code = ''
    all_types = re.findall(type_assigns, file_content)

    imported_modules = defaultdict(set)
    types_dict = {}
    for ori_type in all_types:
        scope, type_name = _parse_type_name(ori_type)
        module = _get_module(type_name, all_modules, scope)
        if not _should_skip(type_name, module):
            imported_modules[module.__module__].add(type_name)
            types_dict[ori_type] = ori_type
            continue
        real_type_assigns = f"{type_name} = '(.*?)'\n"
        matched = re.search(real_type_assigns, file_content)
        if matched is not None:
            real_type = matched.group(1)
            scope, real_type_name = _parse_type_name(real_type)
            module = _get_module(real_type_name, all_modules, scope)
            if not _should_skip(real_type_name, module):
                types_dict[ori_type] = real_type
                imported_modules[module.__module__].add(real_type_name)
        else:
            continue

    for module_name, import_list in imported_modules.items():
        imported_code += f"from {module_name} import {', '.join(import_list)}\n"
    file_content = imported_code + file_content

    for ori_type, real_type in types_dict.items():
        real_scope, real_type_name = _parse_type_name(real_type)
        if not _should_skip(
                real_type_name,
                _get_module(real_type_name, all_modules, real_scope)):
            file_content = re.sub(rf"type='?{ori_type}'?",
                                  f'type={real_type_name}', file_content)

    return file_content


def import_base_modules(filename, file_content):
    basefiles = Config._get_base_files(filename)
    base_modules = []
    for basefile in basefiles:
        scope = None
        if '::' in basefile:
            scope, basefile = basefile.split('::')
        basefile, _ = osp.splitext(basefile)
        if scope is not None:
            base_module = f'{scope}.' + re.sub(r'[-|\.|+]', '_', basefile)
            base_module = base_module.replace('/', '.')
        else:
            no_level = basefile.lstrip('./')
            no_level = re.sub(r'[-|\.|+]', '_', no_level)
            level = len(re.findall(r'\.{2}', basefile)) + 1
            base_module = level * '.' + no_level.replace('/', '.')
        base_modules.append(base_module)

    base_code = ''
    if base_modules:
        base_code = "if '_base_':\n"
        indent = ' ' * 4
        for base_module in base_modules:
            base_code += f'{indent}from {base_module} import *\n'

    removed_code = re.findall(r"(_base_\s=\s\[.*?\]\n|_base_\s=\s'.*?'.*?\n)",
                              file_content, re.DOTALL)
    if removed_code:
        removed_code = removed_code[0]
        file_content = file_content.replace(removed_code, '')
    return base_code + file_content


def resolve_special_syntax(filename, file_content):

    def get_end(start):
        num_pos_para = 0
        invoke = False
        while True:
            s = file_content[start]
            if start == len(file_content) - 1:
                return start
            if s == '(':
                if num_pos_para == 0:
                    invoke = True
                num_pos_para += 1
            elif s == ')':
                num_pos_para -= 1
            elif s == '#':
                while True:
                    start += 1
                    if start == len(file_content) - 1:
                        return start
                    if file_content[start] == '\n':
                        break
            if num_pos_para == 0 and invoke:
                return start
            start += 1

    # refactor inherit logic
    basefiles = Config._get_base_files(filename)
    base_cfgs = dict()
    for basefile in basefiles:
        basefile, _ = Config._get_cfg_path(basefile, filename)
        base_cfgs.update(Config.fromfile(basefile)._cfg_dict)
    for base_key in base_cfgs:
        replaced_string = f'{base_key} = dict'
        matched = re.search(r'\n' + f'({replaced_string})', file_content)
        if matched is None:
            continue
        start = matched.start(1)
        # skip processing comment
        # if matched.groups()[0].startswith('#'):
        #     continue
        # else:
        #     start = matched.start(2)
        end = get_end(start)
        matched_content = file_content[start:end + 1]
        dict_content = matched_content[len(f'{base_key} = '):]
        if isinstance(base_cfgs[base_key], dict):
            file_content = file_content.replace(
                matched_content, f'{base_key}.merge({dict_content})')
        else:
            dict_content = re.sub(r'\_delete\_=True,[\s\n]', '', dict_content)
            file_content = file_content.replace(
                matched_content, f'{base_key} = {dict_content}')
    # remove _base_ in _base_.xxx.
    # file_content = re.sub(r'(?:\{\{)?(?<!\.)_base_\.([\w\._]*)(?:\}\})?', r'\1', file_content)
    file_content = re.sub(
        r"(?:\{\{)?(?<!\.)_base_(?:\.([\w\._]*)(?:\}\})?|\['([\w\._]*)'\])",
        r'\1\2', file_content)
    return file_content


def validate(origin_file, target_file, all_modules):

    def check(origin, target):
        if isinstance(origin, dict):
            assert len(origin) == len(target)
            for k, v in origin.items():
                try:
                    check(v, target[k])
                except Exception as e:
                    logger.debug(
                        f'{k}: {v} != \n{target[k]} in {origin_file} and ./{target_file}'
                    )
                    raise e
        elif isinstance(origin, list):
            assert len(origin) == len(target)
            for item_a, item_b in zip(origin, target):
                try:
                    check(item_a, item_b)
                except Exception as e:
                    logger.debug(
                        f'{item_a} != \n{item_b} in {origin_file} and ./{target_file}'
                    )
                    raise e
        else:
            if origin != target:
                scope, type_name = _parse_type_name(origin)
                module = _get_module(type_name, all_modules, scope)
                assert module == target

    ori_cfg = Config.fromfile(origin_file)._cfg_dict
    target_cfg = Config.fromfile(target_file, lazy_import=True)._cfg_dict

    check(ori_cfg, target_cfg)


def convert_file(filepath, all_modules, target_filepath):
    with open(filepath, 'r') as f:
        file_content = f.read()
        file_content = replace_type_assign(file_content, all_modules)
        file_content = import_base_modules(filepath, file_content)
        file_content = resolve_special_syntax(filepath, file_content)
        try:
            yapf_style = dict(
                based_on_style='pep8',
                blank_line_before_nested_class_or_def=True,
                split_before_expression_after_opening_paren=True)
            file_content, _ = FormatCode(
                file_content, style_config=yapf_style, verify=True)
        except Exception as e:
            with open('debug.py', 'w') as f:
                f.write(file_content)
            raise e
        with open(target_filepath, 'w') as f:
            f.write(file_content)


def convert_files(args):

    def get_target_filepath(filepath):
        target_filepath = filepath.replace(CONFIG_DIR, TARGET_DIR)
        relative_path = osp.relpath(target_filepath, TARGET_DIR)
        name, prefix = osp.splitext(relative_path)
        name = re.sub(r'[-|\.+]', '_', name)
        relative_path = name + prefix
        target_filepath = osp.join(TARGET_DIR, relative_path)
        return target_filepath

    def try_convert_file(target_filepath):
        dirname = osp.dirname(target_filepath)
        mkdir_or_exist(dirname)
        if '__init__.py' not in os.listdir(dirname):
            with open(osp.join(dirname, '__init__.py'), 'w') as f:
                f.write('')
        try:
            convert_file(filepath, all_modules, target_filepath)
        except Exception as e:
            print(f'failed to convert file {filepath}')

    all_modules = collect_all_modules()
    all_files = glob.glob(f'{CONFIG_DIR}/**/*.py', recursive=True)
    target_files = []

    for filepath in track(all_files):
        target_filepath = get_target_filepath(filepath)
        # We could not do the convertion and validation at the same time
        # Since config file is not independent. Config file cannot be validated
        # if its base config is not converted.
        target_files.append(target_filepath)
        if args.convert == 'all':
            try_convert_file(target_filepath)
        elif args.convert == 'none':
            continue
        else:
            if filepath.endswith(args.convert):
                try_convert_file(target_filepath)

    for file, target_file in track(list(zip(all_files, target_files))):
        try:
            if args.validate == 'all':
                validate(file, target_file, all_modules)
            elif args.validate == 'none':
                continue
            else:
                if file.endswith(args.validate):
                    validate(file, target_file, all_modules)
        except Exception as e:
            print(f'failed to validate file {file} and {target_file} for {e}'
                  f'see error information in work_dirs/debug.log')
            # raise e


def parse():
    parser = argparse.ArgumentParser(description='Convert and validate config')
    parser.add_argument(
        '--convert', default='all', type=str, help='convert config file')
    parser.add_argument(
        '--validate', default='all', type=str, help='convert config file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    mmpretrain_register_all_modules()
    convert_files(args)
    # convert_file('configs/gfl/gfl_r101-dconv-c3-c5_fpn_ms-2x_coco.py',
    #              all_modules)
