"""Download pinned public source files for audit; never execute upstream code."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'skeleton-baseline-audit'})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()

records = []
for name, repo in [('protas', 'Yuhan-Shen/ProTAS'), ('mstcn', 'yabufarha/ms-tcn'), ('dptm', 'zczheng777/DPTM')]:
    folder = ROOT / 'sources' / name
    folder.mkdir(parents=True, exist_ok=True)
    meta = json.loads(fetch(f'https://api.github.com/repos/{repo}'))
    branch = meta['default_branch']
    commit = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
    tree = json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))
    if tree.get('truncated'):
        raise RuntimeError(f'Incomplete tree: {repo}')
    (folder / 'tree.json').write_text(json.dumps(tree, indent=2), encoding='utf-8')
    entries = []
    for row in tree['tree']:
        path = row['path']
        if row['type'] != 'blob' or not (path.endswith('.py') or Path(path).name.lower() in ['readme.md', 'license', 'license.md', 'requirements.txt']):
            continue
        content = fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}')
        target = folder / 'upstream' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append(dict(path=path, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    record = dict(name=name, repository=f'https://github.com/{repo}', commit=commit, license=meta.get('license'), files=entries)
    (folder / 'manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    records.append(record)
    print(f'{name}: {commit}, {len(entries)} files', flush=True)
(ROOT / 'sources' / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
