import argparse,json
from pathlib import Path
import yaml
from gcr_llm.config import ExperimentConfig
from gcr_llm.runner import run_dataset

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--config",default="configs/ablation.yaml"); p.add_argument("--output-dir",default="results/raw/ablations"); a=p.parse_args()
    raw=yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    base=ExperimentConfig(seed=raw.get("seed",42),**raw["base"]); outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True); manifest=[]
    for v in raw.get("variants",[]):
        vid=v["id"]; cfg=base.merged({k:x for k,x in v.items() if k!="id"}); method=cfg.ranking.get("method","pagerank"); path=outdir/f"{Path(a.input).stem}_{vid}.json"
        rows=run_dataset(a.input,path,cfg,method); manifest.append({"variant":vid,"output":str(path),"records":len(rows)}); print(vid,len(rows),path)
    (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
if __name__=="__main__": main()
