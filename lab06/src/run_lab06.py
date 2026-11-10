#!/usr/bin/env python3
"""Lab 06 experiments. Run from the repository root: python lab06/src/run_lab06.py."""
import matplotlib
matplotlib.use('Agg')

def main():
    import os, time, random, copy, json, gc, pickle
    from pathlib import Path
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, psutil, yaml, mlflow
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split, Subset
    from torchvision import datasets, transforms
    from torch.profiler import profile, ProfilerActivity, record_function
    def display(value): print(value.to_string(index=False) if hasattr(value, 'to_string') else value)
    ROOT = Path(__file__).resolve().parents[1] / 'results'
    ROOT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config.yaml').read_text(encoding='utf-8'))
    (ROOT/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    mlflow.set_tracking_uri((ROOT/'mlruns').as_uri())
    mlflow.set_experiment(config['experiment'])
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE, '| Output:', ROOT)
    def seed_all(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    seed_all(config['seed'])
    train_all = datasets.FashionMNIST(str(ROOT/'data'), train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]))
    test_data = datasets.FashionMNIST(str(ROOT/'data'), train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]))
    train_data, val_data = random_split(train_all, [55000, 5000], generator=torch.Generator().manual_seed(42))
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); w=config['width']
            self.features=nn.Sequential(nn.Conv2d(1,w,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(w,2*w,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten())
            self.head=nn.Linear(2*w*7*7,10)
        def forward(self,x): return self.head(self.features(x))
    def loader(ds,batch,workers,shuffle=False,seed=None):
        return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=DEVICE.type=='cuda',generator=torch.Generator().manual_seed(config['seed'] if seed is None else seed))
    def sync():
        if DEVICE.type=='cuda': torch.cuda.synchronize()
    def accuracy(model, ds=val_data):
        model.eval(); correct=total=0
        with torch.no_grad():
            for x,y in loader(ds,512,config['workers']):
                x,y=x.to(DEVICE),y.to(DEVICE); correct+=(model(x).argmax(1)==y).sum().item(); total+=len(y)
        return correct/total


    def make_optimizer(model, kind):
        if kind=='Adam': return torch.optim.Adam(model.parameters(),lr=config['adam_lr'])
        return torch.optim.SGD(model.parameters(),lr=config['sgd_lr'],momentum=0.9)
    def make_scheduler(opt, kind, epochs):
        if kind=='StepLR': return torch.optim.lr_scheduler.StepLR(opt,step_size=3,gamma=0.5)
        if kind=='CosineAnnealingLR': return torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=1)
    def train_run(name, epochs=5, batch=128, optimizer='Adam', schedule='StepLR', stop_after=None, resume=None):
        seed_all(config['seed']); model=Net().to(DEVICE); opt=make_optimizer(model,optimizer); scheduler=make_scheduler(opt,schedule,epochs)
        history=[]; start_epoch=0
        if resume:
            ck=torch.load(resume,map_location=DEVICE,weights_only=False)
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
            history=ck['history']; start_epoch=ck['epoch']; random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'].cpu())
            if DEVICE.type=='cuda' and ck['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
        out=ROOT/name; out.mkdir(exist_ok=True); checkpoint=out/'checkpoint.pt'
        with mlflow.start_run(run_name=name):
            mlflow.log_params({'student':config['student'],'seed':config['seed'],'width':config['width'],'batch':batch,'optimizer':optimizer,'scheduler':schedule,'planned_epochs':epochs,'workers':config['workers'],'resume':bool(resume)})
            mlflow.log_artifact(str(ROOT/'config.yaml'))
            for epoch in range(start_epoch, min(epochs, stop_after or epochs)):
                model.train(); total_loss=seen=0; start=time.perf_counter()
                for x,y in loader(train_data,batch,config['workers'],True,config['seed']+epoch):
                    x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(model(x),y); loss.backward(); opt.step()
                    total_loss+=loss.item()*len(y); seen+=len(y)
                sync(); elapsed=time.perf_counter()-start; val=accuracy(model); lr=opt.param_groups[0]['lr']
                row={'epoch':epoch+1,'loss':total_loss/seen,'val_accuracy':val,'samples_per_second':seen/elapsed,'lr':lr}; history.append(row)
                mlflow.log_metrics({k:v for k,v in row.items() if k!='epoch'},step=epoch+1)
                if schedule=='ReduceLROnPlateau': scheduler.step(val)
                else: scheduler.step()
                torch.save({'epoch':epoch+1,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'history':history,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None},checkpoint)
                print(name, row)
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            mlflow.log_artifact(str(checkpoint)); mlflow.log_artifact(str(out/'history.csv'))
        return model, pd.DataFrame(history), checkpoint
    baseline, baseline_history, baseline_ckpt = train_run('baseline', epochs=5)
    print('Baseline held-out test accuracy:', accuracy(baseline,test_data))


    seed_all(config['seed']); pmodel=Net().to(DEVICE); popt=make_optimizer(pmodel,'Adam'); pmodel.train()
    times=dict.fromkeys(['data_loading','forward','backward','optimizer'],0.0)
    activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if DEVICE.type=='cuda' else [])
    with profile(activities=activities,record_shapes=True) as prof:
        iterator=iter(loader(train_data,128,config['workers'],True))
        while True:
            sync(); t=time.perf_counter()
            try:
                with record_function('data_loading'): x,y=next(iterator); x,y=x.to(DEVICE),y.to(DEVICE)
            except StopIteration: break
            sync(); times['data_loading']+=time.perf_counter()-t
            t=time.perf_counter()
            with record_function('forward'): loss=nn.functional.cross_entropy(pmodel(x),y)
            sync(); times['forward']+=time.perf_counter()-t
            t=time.perf_counter()
            with record_function('backward'): popt.zero_grad(set_to_none=True); loss.backward()
            sync(); times['backward']+=time.perf_counter()-t
            t=time.perf_counter()
            with record_function('optimizer'): popt.step()
            sync(); times['optimizer']+=time.perf_counter()-t
    prof.export_chrome_trace(str(ROOT/'profile_trace.json'))
    profile_table=pd.DataFrame({'region':list(times),'seconds':list(times.values())}); profile_table['share_percent']=100*profile_table.seconds/profile_table.seconds.sum(); display(profile_table)
    profile_table.to_csv(ROOT/'profile_breakdown.csv',index=False)
    with mlflow.start_run(run_name='profile'): mlflow.log_artifacts(str(ROOT), artifact_path='profile_files') if False else mlflow.log_artifact(str(ROOT/'profile_breakdown.csv')); mlflow.log_artifact(str(ROOT/'profile_trace.json'))


    worker_results=[]
    for workers in [0,1,2,4,8]:
        seed_all(config['seed']); m=Net().to(DEVICE); o=make_optimizer(m,'Adam'); m.train(); count=0; sync(); start=time.perf_counter()
        for x,y in loader(train_data,128,workers,True):
            x,y=x.to(DEVICE),y.to(DEVICE); o.zero_grad(set_to_none=True); nn.functional.cross_entropy(m(x),y).backward(); o.step(); count+=len(y)
        sync(); worker_results.append({'workers':workers,'samples_per_second':count/(time.perf_counter()-start)})
    worker_df=pd.DataFrame(worker_results); display(worker_df); worker_df.to_csv(ROOT/'workers.csv',index=False)
    worker_df.plot(x='workers',y='samples_per_second',marker='o',grid=True,title='DataLoader workers vs training throughput'); plt.ylabel('Samples / second'); plt.tight_layout(); plt.savefig(ROOT/'workers.png'); plt.show()
    print('Fastest observed worker count:',int(worker_df.loc[worker_df.samples_per_second.idxmax(),'workers']))
    with mlflow.start_run(run_name='workers'): mlflow.log_artifact(str(ROOT/'workers.csv')); mlflow.log_artifact(str(ROOT/'workers.png'))


    batch_results=[]
    for b in [32,64,128,256,512]:
        gc.collect()
        if DEVICE.type=='cuda': torch.cuda.reset_peak_memory_stats()
        m,h,cp=train_run('batch_'+str(b),epochs=5,batch=b)
        if DEVICE.type=='cuda': mem=torch.cuda.max_memory_allocated()/1024**2; mem_kind='GPU allocated peak MiB'
        else:
            import resource
            mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024; mem_kind='process high-water RSS MiB'
        batch_results.append({'batch':b,'samples_per_second':h.samples_per_second.mean(),'memory_MiB':mem,'memory_type':mem_kind,'final_val_accuracy':h.val_accuracy.iloc[-1]})
    batch_df=pd.DataFrame(batch_results); display(batch_df); batch_df.to_csv(ROOT/'batch_study.csv',index=False)
    with mlflow.start_run(run_name='batch_summary'): mlflow.log_artifact(str(ROOT/'batch_study.csv'))


    comparison=[]
    for opt in ['SGD','Adam']:
        for sched in ['StepLR','CosineAnnealingLR','ReduceLROnPlateau']:
            _,h,_=train_run(opt+'_'+sched,epochs=10,batch=128,optimizer=opt,schedule=sched)
            comparison.append({'optimizer':opt,'schedule':sched,'final_val_accuracy':h.val_accuracy.iloc[-1],'best_val_accuracy':h.val_accuracy.max()})
            plt.plot(h.epoch,h.val_accuracy,label=opt+' / '+sched)
    compare_df=pd.DataFrame(comparison); display(compare_df); compare_df.to_csv(ROOT/'schedules.csv',index=False)
    plt.xlabel('Epoch'); plt.ylabel('Validation accuracy'); plt.legend(fontsize=8); plt.grid(); plt.tight_layout(); plt.savefig(ROOT/'schedules.png'); plt.show()
    with mlflow.start_run(run_name='schedule_summary'): mlflow.log_artifact(str(ROOT/'schedules.csv')); mlflow.log_artifact(str(ROOT/'schedules.png'))


    seed_all(config['seed']); initial=Net().state_dict(); subset=Subset(train_data,range(4096))
    def accumulation_test(batch,accum):
        m=Net().to(DEVICE); m.load_state_dict(copy.deepcopy(initial)); opt=torch.optim.SGD(m.parameters(),lr=0.01); m.train(); losses=[]
        dl=loader(subset,batch,0,False); opt.zero_grad(set_to_none=True)
        for i,(x,y) in enumerate(dl):
            group_start=(i//accum)*accum*batch; group_n=min(accum*batch,len(subset)-group_start)
            x,y=x.to(DEVICE),y.to(DEVICE); loss=nn.functional.cross_entropy(m(x),y); (loss*len(y)/group_n).backward(); losses.append(loss.item())
            if (i+1)%accum==0 or i+1==len(dl): opt.step(); opt.zero_grad(set_to_none=True)
        return m, np.mean(losses)
    a,la=accumulation_test(256,1); b,lb=accumulation_test(64,4)
    diff=max((x-y).abs().max().item() for x,y in zip(a.parameters(),b.parameters()))
    accum_result={'batch256_mean_loss':la,'accum64_mean_loss':lb,'max_parameter_difference':diff,'batch256_val_accuracy':accuracy(a),'accum_val_accuracy':accuracy(b)}
    print(accum_result)
    (ROOT/'accumulation.json').write_text(json.dumps(accum_result,indent=2))
    with mlflow.start_run(run_name='accumulation'): mlflow.log_metrics(accum_result); mlflow.log_artifact(str(ROOT/'accumulation.json'))


    _,full,_=train_run('continuous',epochs=5)
    _,part,ck=train_run('interrupted',epochs=5,stop_after=3)
    _,resumed,_=train_run('resumed',epochs=5,resume=ck)
    check=full.merge(resumed,on='epoch',suffixes=('_continuous','_resumed')); check['accuracy_difference']=(check.val_accuracy_continuous-check.val_accuracy_resumed).abs(); display(check[['epoch','val_accuracy_continuous','val_accuracy_resumed','accuracy_difference']]); print('Maximum accuracy difference:',check.accuracy_difference.max())
    plt.plot(full.epoch,full.val_accuracy,'o-',label='Continuous'); plt.plot(resumed.epoch,resumed.val_accuracy,'x--',label='Resumed'); plt.axvline(3,color='gray',linestyle=':'); plt.xlabel('Epoch'); plt.ylabel('Validation accuracy'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(ROOT/'resume_comparison.png'); plt.show()
    check.to_csv(ROOT/'resume_comparison.csv',index=False)
    with mlflow.start_run(run_name='resume_comparison'): mlflow.log_artifact(str(ROOT/'resume_comparison.csv')); mlflow.log_artifact(str(ROOT/'resume_comparison.png'))
    print('All generated files:',ROOT); print('To inspect tracking: mlflow ui --backend-store-uri', (ROOT/'mlruns').as_uri())


if __name__ == '__main__':
    main()
