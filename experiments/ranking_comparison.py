import argparse,json
from pathlib import Path
from gcr_llm.config import ExperimentConfig
from gcr_llm.runner import run_dataset

METHODS=["single_model","independent_multi_sample","direct_similarity","majority","pagerank","eigenvector"]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--config",default="configs/graph_reasoning.yaml"); p.add_argument("--output",default="results/raw/ranking_comparison.json"); a=p.parse_args()
    cfg=ExperimentConfig.from_yaml(a.config); rows=[]
    for method in METHODS:
        tmp=Path(a.output).with_suffix(f".{method}.json"); rows.extend(run_dataset(a.input,tmp,cfg,method))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(rows,indent=2),encoding="utf-8"); print("Saved",a.output)
if __name__=="__main__": main()
