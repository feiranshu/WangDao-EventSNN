"""Original floating H2-pro ANN; original per-channel int8 export, no SNN/QAT."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import FLOAT_SOURCE,INT8_SOURCE,sha,ANN_ROOT
from wangdao.h2.export import quantize_weight


class Refiner(nn.Module):
    def __init__(self,source=FLOAT_SOURCE):
        super().__init__()
        self.old=nn.Linear(35,64,bias=False)
        self.new=nn.Linear(1,64,bias=False)
        self.hidden=nn.Linear(64,32,bias=False)
        self.output=nn.Linear(32,4,bias=False)
        checkpoint=torch.load(source,map_location='cpu',weights_only=True)
        if checkpoint['protocol_sha256']!=sha(ANN_ROOT/'software/wangdao/h2_pro/protocol.json'):
            raise ValueError('Original ANN feature protocol mismatch')
        self.load_state_dict(checkpoint['state_dict'],strict=True)

    def forward(self,features):
        if features.dtype!=torch.uint8 or features.ndim!=2 or features.shape[1]!=36:
            raise ValueError('Require original 36 uint8 features')
        x=features.float()/255.
        other=torch.cat((x[:,:2],x[:,3:]),dim=1)
        x=F.hardtanh(self.old(other)+self.new(x[:,2:3]),0.,1.)
        x=F.hardtanh(self.hidden(x),0.,1.)
        return self.output(x)

    def combined(self):
        return (torch.cat((self.old.weight[:,:2],self.new.weight,self.old.weight[:,2:]),dim=1),
                self.hidden.weight,self.output.weight)

    def export_layers(self):return tuple(quantize_weight(w) for w in self.combined())


def logits(raw):return raw.reshape(-1,2,2).amax(dim=2)


def require_finite(values,where):
    values=list(values)
    if values and not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):
        raise FloatingPointError('Nonfinite '+where+'; last committed checkpoint retained')


def optimizer_for(model,config):
    return torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,
                            betas=(.9,.999),eps=1e-8,foreach=False)


def supervised(model,x,y):
    raw=model(x)
    loss=F.cross_entropy(logits(raw),y.long()) if len(x) else sum(p.sum()*0 for p in model.parameters())
    require_finite([raw,loss],'logits/loss')
    return raw,loss


def update(model,optimizer,x,y):
    model.train();optimizer.zero_grad(set_to_none=True)
    raw,loss=supervised(model,x,y);loss.backward()
    gradients=[p.grad for p in model.parameters()]
    if any(g is None for g in gradients):raise RuntimeError('Missing original matrix gradient')
    require_finite(gradients,'gradients')
    norm=torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients]))
    optimizer.step();require_finite(model.parameters(),'weights')
    return dict(candidate_events=len(x),loss=float(loss.detach()),gradient_norm=float(norm),
                max_abs_logit=float(raw.detach().abs().max()) if len(raw) else 0.)


def exact_layers_on_device(layers,device):
    return [(torch.from_numpy(w.copy()).to(device,dtype=torch.float64),
             torch.from_numpy(s.copy()).to(device,dtype=torch.float64)) for w,s in layers]


@torch.no_grad()
def exact_forward(x,layers):
    value=x.double()
    for i,(w,s) in enumerate(layers):
        value=(value@w)*s/255.
        if i<2:value=torch.round(value.clamp(0.,1.)*255.)
    return value.float()


def load_export(path):
    with np.load(path,allow_pickle=False) as z:
        if tuple(z['dimensions'])!=(36,64,32,4):raise ValueError('Wrong exported architecture')
        return tuple((z[f'layer{i}_weight'].copy(),z[f'layer{i}_scale'].copy()) for i in range(3))


def assert_initial_export(model):
    frozen=load_export(INT8_SOURCE)
    for (w,s),(ew,es) in zip(model.export_layers(),frozen):
        np.testing.assert_array_equal(w,ew);np.testing.assert_array_equal(s,es)
    return True
