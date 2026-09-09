"""Pure continuation policies, shared by training and CPU acceptance checks."""
import math

def stage_factor(elapsed,duration,warm,floor):
    assert 0<warm<duration and 0<floor<1
    if elapsed<warm:return .1+.9*max(0,elapsed)/warm
    t=min(1.,max(0.,(elapsed-warm)/(duration-warm)))
    return floor+(1-floor)*.5*(1+math.cos(math.pi*t))

def check_parent_evaluation(parent,sets):
    differences={}
    for group in ['train_probe','val_v2','val_v6']:
        old={r['scene']:r for r in parent['sets'][group]['scenes']};new={r['scene']:r for r in sets[group]['scenes']}
        assert old.keys()==new.keys(),f'{group}: scene membership changed'
        for key in old:
            assert old[key]['context']==new[key]['context'] and old[key]['targets']==new[key]['targets'],f'{group}: evaluation views changed'
        differences[group]={metric:max(abs(old[k][metric]-new[k][metric]) for k in old) for metric in ['psnr','lpips','total']}
        assert differences[group]['psnr']<.01 and differences[group]['lpips']<.001 and differences[group]['total']<.001,(group,differences[group])
    return dict(passed=True,maximum_absolute_difference=differences,note='Same checkpoint and fixed evaluation; bounded numeric tolerance, not a bitwise claim')
