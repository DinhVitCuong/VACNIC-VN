# vncore_singleton.py
import py_vncorenlp

_VNCORE = None

def get_vncore(save_dir, with_heap=True):
    global _VNCORE
    py_vncorenlp.download_model(save_dir=save_dir)
    if _VNCORE is None:
        # Lưu ý: annotators cần ["wseg","pos","ner"] để NER chạy đúng
        kwargs = dict(annotators=["wseg","pos","ner"], save_dir=save_dir)
        if with_heap:
            kwargs["max_heap_size"] = "-Xmx10g"  # chỉ gán ở lần đầu
        _VNCORE = py_vncorenlp.VnCoreNLP(**kwargs)
    return _VNCORE
