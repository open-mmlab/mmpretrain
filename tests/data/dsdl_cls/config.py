# Copyright (c) OpenMMLab. All rights reserved.

local = dict(
    type="LocalFileReader",
    working_dir="the root path of the prepared dataset",)

ali_oss = dict(
    type="AliOSSFileReader",
    access_key_secret="your secret key of aliyun oss",
    endpoint="your endpoint of aliyun oss",
    access_key_id="your access key of aliyun oss",
    bucket_name="your bucket name of aliyun oss",
    working_dir="the path of the prepared dataset without the bucket's name")
    