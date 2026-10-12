"""Fixed training-only channel statistics with explicit per-person validity."""
import numpy as np


def feature_validity(raw):
    p = np.asarray(raw).reshape(-1,2,25,3)
    valid = np.any(p != 0,axis=(2,3))
    motion = np.zeros_like(valid)
    motion[1:] = valid[1:] & valid[:-1]
    coordinates = np.repeat(valid,75,axis=1)
    displacement = np.repeat(motion,75,axis=1)
    return np.concatenate([coordinates,displacement],axis=1).T


class TrainingStandardizer:
    def __init__(self,mean,scale,count):
        self.mean = np.asarray(mean,dtype=np.float32)
        self.scale = np.asarray(scale,dtype=np.float32)
        self.count = np.asarray(count,dtype=np.int64)

    @classmethod
    def fit(cls,examples,std_floor=1e-3):
        count = np.zeros(300,dtype=np.int64)
        total = np.zeros(300,dtype=np.float64)
        squares = np.zeros(300,dtype=np.float64)
        for features,valid in examples:
            values = features.astype(np.float64)
            count += valid.sum(1)
            total += np.where(valid,values,0).sum(1)
            squares += np.where(valid,values*values,0).sum(1)
        mean = np.divide(total,count,out=np.zeros(300),where=count>0)
        variance = np.divide(squares,count,out=np.zeros(300),where=count>0)-mean*mean
        scale = np.maximum(np.sqrt(np.maximum(variance,0)),std_floor)
        # No observation => identity parameters; do not invent training statistics.
        scale[count==0] = 1.
        return cls(mean,scale,count)

    def transform(self,features,valid):
        output = np.where(valid,(features-self.mean[:,None])/self.scale[:,None],0)
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite normalized features')
        return output.astype(np.float32)

    def save(self,path):
        np.savez(path,mean=self.mean,scale=self.scale,count=self.count,std_floor=1e-3)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            return cls(data['mean'],data['scale'],data['count'])
