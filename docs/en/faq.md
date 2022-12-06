# Frequently Asked Questions

We list some common troubles faced by many users and their corresponding
solutions here. Feel free to enrich the list if you find any frequent issues
and have ways to help others to solve them. If the contents here do not cover
your issue, please create an issue using the
[provided templates](https://github.com/open-mmlab/mmclassification/issues/new/choose)
and make sure you fill in all required information in the template.

## Installation

- Compatibility issue between MMCV and MMClassification; "AssertionError:
  MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

  Compatible MMClassification and MMCV versions are shown as below. Please
  choose the correct version of MMCV to avoid installation issues.

  | MMClassification version |      MMCV version      |
  | :----------------------: | :--------------------: |
  |           dev            |  mmcv>=1.7.0, \<1.9.0  |
  |     0.25.0 (master)      |  mmcv>=1.4.2, \<1.9.0  |
  |          0.24.1          |  mmcv>=1.4.2, \<1.9.0  |
  |          0.23.2          |  mmcv>=1.4.2, \<1.7.0  |
  |          0.22.1          |  mmcv>=1.4.2, \<1.6.0  |
  |          0.21.0          | mmcv>=1.4.2, \<=1.5.0  |
  |          0.20.1          | mmcv>=1.4.2, \<=1.5.0  |
  |          0.19.0          | mmcv>=1.3.16, \<=1.5.0 |
  |          0.18.0          | mmcv>=1.3.16, \<=1.5.0 |
  |          0.17.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.16.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.15.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.15.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.14.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.13.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.12.0          | mmcv>=1.3.1, \<=1.5.0  |
  |          0.11.1          | mmcv>=1.3.1, \<=1.5.0  |
  |          0.11.0          |      mmcv>=1.3.0       |
  |          0.10.0          |      mmcv>=1.3.0       |
  |          0.9.0           |      mmcv>=1.1.4       |
  |          0.8.0           |      mmcv>=1.1.4       |
  |          0.7.0           |      mmcv>=1.1.4       |
  |          0.6.0           |      mmcv>=1.1.4       |

  ```{note}
  Since the `dev` branch is under frequent development, the MMCV
  version dependency may be inaccurate. If you encounter problems when using
  the `dev` branch, please try to update MMCV to the latest version.
  ```

- Using Albumentations

  If you would like to use `albumentations`, we suggest using `pip install -r requirements/albu.txt` or
  `pip install -U albumentations --no-binary qudida,albumentations`.

  If you simply use `pip install albumentations>=0.3.2`, it will install `opencv-python-headless` simultaneously
  (even though you have already installed `opencv-python`). Please refer to the
  [official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies)
  for details.

## Coding

- Do I need to reinstall mmcls after some code modifications?

  If you follow [the best practice](install.md) and install mmcls from source,
  any local modifications made to the code will take effect without
  reinstallation.

- How to develop with multiple MMClassification versions?

  Generally speaking, we recommend to use different virtual environments to
  manage MMClassification in different working directories. However, you
  can also use the same environment to develop MMClassification in different
  folders, like mmcls-0.21, mmcls-0.23. When you run the train or test shell script,
  it will adopt the mmcls package in the current folder. And when you run other Python
  script, you can also add `` PYTHONPATH=`pwd`  `` at the beginning of your command
  to use the package in the current folder.

  Conversely, to use the default MMClassification installed in the environment
  rather than the one you are working with, you can remove the following line
  in those shell scripts:

  ```shell
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
  ```
