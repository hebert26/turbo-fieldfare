import hashlib, pathlib
BASE=pathlib.Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def overlay_hashes():
    result={}
    for name in ['Package.swift','Package.resolved']:
        result['overlay/'+name]=sha(BASE/'overlay'/name)
    # Complete source/resource inputs plus pinned dependency checkout inputs and binary bridge.
    # Never walks scratch, original shards, model packs or generated build products.
    for label,root in [('overlay/Sources',BASE/'overlay/Sources'),('overlay/ThirdParty',BASE/'overlay/ThirdParty'),('build/checkouts',BASE/'build/checkouts')]:
        for path in sorted(root.rglob('*')):
            if any(part in ['.git','.build'] for part in path.relative_to(root).parts):continue
            if path.is_file():result[label+'/'+path.relative_to(root).as_posix()]=sha(path)
    return result
