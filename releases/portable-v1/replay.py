#!/usr/bin/env python3
"""Portable Wind VISION results replay. No training, checkpoint execution or CUDA."""
import argparse, hashlib, json, platform, statistics, sys
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
BANDS = {'low':(1,5),'middle':(5,17),'high':(17,65),'tail':(65,129)}

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition: raise ValueError(message)

def verify(root=ROOT):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path=(root/name).resolve()
        require(path.is_relative_to(root.resolve()), 'Unsafe manifest path')
        require(path.is_file() and sha256(path)==expected, 'Integrity mismatch: '+name)
    return len(manifest['files_sha256'])

def transfer_weights(xyz, obs, hid):
    distances=np.linalg.norm(xyz[hid,None,:]-xyz[None,obs,:],axis=-1)
    nearest=np.argsort(distances,axis=1)[:,:4]
    weights=1/np.maximum(np.take_along_axis(distances,nearest,axis=1),1e-6)**2
    weights/=weights.sum(axis=1,keepdims=True)
    W=np.zeros((len(hid),len(obs)))
    np.put_along_axis(W,nearest,weights,axis=1)
    return W

def classical_predictions(xyz, observed, obs, hid):
    """Only observed time series and geometry are accepted. Hidden truth is absent."""
    W=transfer_weights(xyz,obs,hid)
    transfer=lambda values: np.einsum('uo,not->nut',W,values,optimize=True)
    heights=np.unique(xyz[:,2])
    oh=np.argmin(abs(xyz[obs,2,None]-heights[None,:]),axis=1)
    hh=np.argmin(abs(xyz[hid,2,None]-heights[None,:]),axis=1)
    profile=np.stack([observed.mean(axis=-1)[:,oh==k].mean(axis=1) for k in range(len(heights))],axis=1)
    mean=2.5*1.9*np.log(xyz[:,2]/0.015)
    return {
        'OBS_ESTIMATED_MEAN_FULL_IDW': profile[:,hh,None]+transfer(observed-profile[:,oh,None]),
        'KNOWN_MEAN_FULL_IDW': mean[hid][None,:,None]+transfer(observed-mean[obs][None,:,None]),
        'DIRECT_WIND_IDW': transfer(observed),
    }

def metrics(pred, truth):
    err=pred-truth
    ref=truth-truth.mean(axis=-1,keepdims=True)
    an=pred-pred.mean(axis=-1,keepdims=True)
    variance=np.mean(ref**2)
    def spec(x):
        y=x-x.mean(axis=-1,keepdims=True)
        return (abs(np.fft.rfft(y*np.hanning(x.shape[-1]),axis=-1))**2/x.shape[-1]**2).mean(axis=(0,1))
    ps,ts=spec(pred),spec(truth)
    mse=np.mean(err**2,axis=(1,2))
    return dict(unobserved_rmse_mps=float(np.sqrt(mse.mean())),
        per_field_unobserved_rmse_mps=np.sqrt(mse).tolist(),
        unobserved_mae_mps=float(np.mean(abs(err))), unobserved_bias_mps=float(err.mean()),
        temporal_anomaly_r2=float(1-np.mean((an-ref)**2)/variance),
        temporal_variance_ratio=float(np.mean(an**2)/variance),
        interior_step_rms_ratio=float(np.sqrt(np.mean(np.diff(pred,axis=-1)**2)/np.mean(np.diff(truth,axis=-1)**2))),
        endpoint_jump_rms_mps=float(np.sqrt(np.mean((pred[:,:,-1]-pred[:,:,0])**2))),
        hann_band_power_ratios={k:float(ps[a:b].sum()/ts[a:b].sum()) for k,(a,b) in BANDS.items()})

def compare(actual, expected, label, tolerance=2e-6):
    if isinstance(expected,dict):
        for k,v in expected.items(): compare(actual[k],v,label+'.'+k,tolerance)
    else:
        require(np.allclose(actual,expected,atol=tolerance,rtol=0), 'Result mismatch: '+label)

def replay(output):
    count=verify(); print(f'Integrity: PASS ({count} files)',flush=True)
    protocol=json.loads((ROOT/'evidence/protocol.json').read_text())
    original=json.loads((ROOT/'evidence/original_results.json').read_text())
    indices=np.array(protocol['final_indices'])
    obs=np.array(protocol['observed_sensor_indices']); hid=np.array(protocol['heldout_sensor_indices'])
    require(len(indices)==872 and len(set(indices))==872, 'Invalid final split')
    require(len(obs)==40 and len(hid)==60 and not set(obs)&set(hid), 'Invalid sensor partition')
    with h5py.File(ROOT/'data/wind_test_data.hdf5','r') as f:
        xyz=np.asarray(f['x_grid'],dtype=np.float32).astype(np.float64)
        # Match the original experiment's float32 ingestion, then float64 evaluation.
        fields=np.stack([np.asarray(f['wind_samples'][str(int(i))]['v1'],dtype=np.float32) for i in indices]).astype(np.float64)
    observed=fields[:,obs,:]
    truth=fields[:,hid,:]
    del fields
    results={}; mse={}
    with np.load(ROOT/'data/frozen_predictions_872.npz',allow_pickle=False) as frozen:
        require(np.array_equal(frozen['final_indices'],indices), 'Frozen split mismatch')
        require(str(frozen['protocol_sha256'])==sha256(ROOT/'evidence/protocol.json'), 'Frozen protocol mismatch')
        for name, expected in original['results'].items():
            pred=frozen[name.lower()].astype(np.float64)
            require(pred.shape==(872,60,256) and np.isfinite(pred).all(), 'Invalid prediction array')
            results[name]=metrics(pred,truth)
            compare(results[name],expected,name)
            mse[name]=np.array(results[name]['per_field_unobserved_rmse_mps'])**2
            print(f'{name}: RMSE {results[name]["unobserved_rmse_mps"]:.9f} m/s — PASS',flush=True)
        # These regenerate predictions, rather than just evaluate stored arrays.
        regenerated={}
        for name,pred in classical_predictions(xyz,observed,obs,hid).items():
            diff=float(np.max(abs(pred-frozen[name.lower()].astype(np.float64))))
            require(diff<1e-5, 'Regenerated baseline mismatch: '+name)
            regenerated[name]={'max_abs_difference_from_frozen_mps':diff}
    boots=np.random.default_rng(protocol['bootstrap_seed']).integers(0,872,size=(protocol['bootstrap_resamples'],872))
    primary=protocol['primary_selected_method']
    intervals={}
    for name,expected in original['paired_comparisons_positive_favors_selected'].items():
        gains=np.sqrt(mse[name][boots].mean(axis=1))-np.sqrt(mse[primary][boots].mean(axis=1))
        interval=np.quantile(gains,[.025,.975]).tolist()
        compare(interval,expected['paired_bootstrap_95pct_interval_mps'],name+'.bootstrap',1e-9)
        intervals[name]=interval
    benchmark=json.loads((ROOT/'evidence/four_arm_benchmark.json').read_text())
    medians={k:statistics.median(v) for k,v in benchmark['raw_step_times_ms'].items()}
    compare(medians,benchmark['median_step_times_ms'],'benchmark',1e-9)
    speedup=medians['A_original_pytorch']/medians['D_optimized_physicsnemo']
    print('Three classical baselines regenerated; seven paired bootstrap intervals matched; four timing medians matched.',flush=True)
    receipt=dict(status='PASS',scope='CPU results replay, not neural retraining or a new GPU benchmark',
        verified_files=count,final_fields=872,hidden_targets=int(truth.size),methods=results,
        regenerated_baselines=regenerated,bootstrap_resamples=protocol['bootstrap_resamples'],
        paired_bootstrap_95pct_intervals_mps=intervals,benchmark_medians_ms=medians,benchmark_speedup_from_saved_timings=speedup,
        environment=dict(python=platform.python_version(),numpy=np.__version__,h5py=h5py.__version__,platform=platform.system()),
        gpu_experiments_rerun=False,neural_predictions_regenerated=False,
        limitations=['Single simulated velocity component and fixed geometry; spatial reconstruction, not forecasting.',
            'Frozen neural predictions are rescored. They are not regenerated on CPU.',
            'Historical CUDA timings and parity checks are archived evidence, not rerun measurements.',
            'No claim of flight qualification or operational weather validation.'])
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS — receipt: '+str(output),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('replay-receipt.json'))
    args=parser.parse_args()
    try:
        if args.verify_only: print(f'PASS: {verify()} SHA-256 checks')
        else: replay(args.output)
    except (ValueError,OSError,KeyError) as error:
        print('FAIL: '+str(error),file=sys.stderr);sys.exit(1)
