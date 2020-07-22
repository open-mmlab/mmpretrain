import gzip
import hashlib
import os
import os.path
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile

__all__ = ['rm_suffix', 'check_integrity', 'download_and_extract_archive']


def rm_suffix(s, suffix=None):
    if suffix is None:
        return s[:s.rfind('.')]
    else:
        return s[:s.rfind(suffix)]


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url_to_file(url, fpath):
    with urllib.request.urlopen(url) as resp, open(fpath, 'wb') as of:
        shutil.copyfileobj(resp, of)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from.
        root (str): Directory to place downloaded file in.
        filename (str | None): Name to save the file under.
            If filename is None, use the basename of the URL.
        md5 (str | None): MD5 checksum of the download.
            If md5 is None, download without md5 check.
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f'Using downloaded and verified file: {fpath}')
    else:
        try:
            print(f'Downloading {url} to {fpath}')
            download_url_to_file(url, fpath)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      f' Downloading {url} to {fpath}')
                download_url_to_file(url, fpath)
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError('File not found or corrupted.')


def _is_tarxz(filename):
    return filename.endswith('.tar.xz')


def _is_tar(filename):
    return filename.endswith('.tar')


def _is_targz(filename):
    return filename.endswith('.tar.gz')


def _is_tgz(filename):
    return filename.endswith('.tgz')


def _is_gzip(filename):
    return filename.endswith('.gz') and not filename.endswith('.tar.gz')


def _is_zip(filename):
    return filename.endswith('.zip')


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path,
            os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, 'wb') as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError(f'Extraction of {from_path} not supported')

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url,
                                 download_root,
                                 extract_root=None,
                                 filename=None,
                                 md5=None,
                                 remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f'Extracting {archive} to {extract_root}')
    extract_archive(archive, extract_root, remove_finished)
