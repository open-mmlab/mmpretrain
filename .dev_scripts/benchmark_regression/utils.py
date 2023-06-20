from pathlib import Path

HTTP_PREFIX = 'https://download.openmmlab.com/'
MMCLS_ROOT = Path(__file__).absolute().parents[2]
METRICS_MAP = {
    'Top 1 Accuracy': 'accuracy/top1',
    'Top 5 Accuracy': 'accuracy/top5',
    'Recall@1': 'retrieval/Recall@1',
    'Recall@5': 'retrieval/Recall@5',
    'BLEU-4': 'Bleu_4',
    'CIDER': 'CIDEr',
}


def substitute_weights(download_link, root):
    if 's3://' in root:
        from mmengine.fileio.backends import PetrelBackend
        from petrel_client.common.exception import AccessDeniedError
        file_backend = PetrelBackend()
        checkpoint = file_backend.join_path(root,
                                            download_link[len(HTTP_PREFIX):])
        try:
            exists = file_backend.exists(checkpoint)
        except AccessDeniedError:
            exists = False
    else:
        checkpoint = Path(root) / download_link[len(HTTP_PREFIX):]
        exists = checkpoint.exists()

    if exists:
        return str(checkpoint)
    else:
        return None
