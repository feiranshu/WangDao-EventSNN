"""Original r1 numerics with shared projection and equally weighted prefix loss."""
import torch
from ..h2pro_r1_train_v1.model import R1T16, RoundSTE, losses, require_finite, update as original_update
from .common import STEPS


class PrefixCountSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, steps):
        if steps not in STEPS: raise ValueError('Only frozen 4/8/16 prefixes')
        ctx.steps=steps
        return torch.floor((activation*steps+191)/255)

    @staticmethod
    def backward(ctx, gradient):
        return gradient*(ctx.steps/255), None


class R1Multi(R1T16):
    """Identical state_dict/parameters; inherited forward is unchanged T16."""
    def prefixes(self,features):
        if features.dtype!=torch.uint8 or features.ndim!=2 or features.shape[1]!=36:
            raise ValueError('Original uint8 36-column features required')
        weights=[self.integer_weight(i) for i in range(3)]
        scales=[getattr(self,f'scale{i}').double() for i in range(3)]
        h1=RoundSTE.apply(((features.double()@weights[0])*scales[0]/255).clamp(0,1)*255)
        raw,spikes=[],[]
        for t in STEPS:
            count=PrefixCountSTE.apply(h1,t)
            h2=RoundSTE.apply(((count@weights[1])*scales[1]/t).clamp(0,1)*255)
            raw.append(((h2@weights[2])*scales[2]/255).float())
            spikes.append(count.detach().sum(dim=1).to(torch.int64))
        return torch.stack(raw),torch.stack(spikes)


def joint_losses(raw,labels,teacher,config):
    if raw.ndim!=3 or raw.shape[0]!=3 or raw.shape[2]!=4:
        raise ValueError('Expected original outputs at T4/T8/T16')
    per_prefix=torch.stack([torch.stack(losses(r,labels,teacher,config)) for r in raw])
    return per_prefix.mean(dim=0),per_prefix


def update(model,optimizer,batch,config,arm):
    if arm=='control':
        result=original_update(model,optimizer,batch,config)
        result.update(prefix_losses={'16':result['losses']},prefix_spikes={'16':result['spikes'][0]})
        return result
    if arm!='multit': raise ValueError('Unknown training arm')
    model.train();optimizer.zero_grad(set_to_none=True)
    raw,spikes=model.prefixes(batch['features'])
    terms,per_prefix=joint_losses(raw,batch['labels'],batch['teacher'],config)
    require_finite([raw,terms,per_prefix],'prefix outputs/loss')
    terms[0].backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if len(gradients)!=3: raise RuntimeError('Every original matrix must receive a gradient')
    require_finite(gradients,'gradients')
    norm=torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients]))
    optimizer.step();require_finite(model.parameters(),'student weights')
    summed=spikes.sum(dim=1).cpu().tolist()
    return dict(losses=terms.detach().cpu().tolist(),gradient_norm=float(norm),
                logit_absmax=float(raw.detach().abs().max()),spikes=[summed[2],0],events=raw.shape[1],
                prefix_losses={str(t):v for t,v in zip(STEPS,per_prefix.detach().cpu().tolist())},
                prefix_spikes=dict(zip(map(str,STEPS),summed)))
