"""Verify statistics and masking before training the normalization control."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import skeleton_features
from normalization import TrainingStandardizer,feature_validity
rng=np.random.default_rng(17)
raw=rng.normal(size=(20,150)).astype(np.float32)
raw[:,75:]=0
raw[5,:75]=0
x=skeleton_features(raw);valid=feature_validity(raw)
standardizer=TrainingStandardizer.fit([(x,valid)])
out=standardizer.transform(x,valid)
assert (out[~valid]==0).all()
assert (standardizer.count[75:150]==0).all() and (standardizer.scale[75:150]==1).all()
for channel in np.flatnonzero(standardizer.count):
    values=out[channel,valid[channel]]
    assert abs(values.mean())<1e-5
    if x[channel,valid[channel]].std()>1e-3:
        assert abs(values.std()-1)<1e-5
altered=raw.copy();altered[10:]=rng.normal(size=(10,150))*50
altered_x=skeleton_features(altered)
altered_valid=feature_validity(altered)
assert np.array_equal(standardizer.transform(altered_x,altered_valid)[:,:10],out[:,:10])
assert not out[150:,0].any()
assert not out[150:225,5:7].any()
path=ROOT/'reports/normalization_check_stats.npz'
standardizer.save(path)
loaded=TrainingStandardizer.load(path)
assert np.array_equal(loaded.transform(x,valid),out)
result=dict(passed=True,valid_channel_center_and_scale=True,missing_body_stays_zero=True,
            displacement_gap_stays_zero=True,fixed_statistics_future_invariance=True,serialization_roundtrip=True,
            note='Training entry fits selected train list only; no clipping; floor=1e-3; unseen channels use mean0,scale1.')
(ROOT/'reports/normalization_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
