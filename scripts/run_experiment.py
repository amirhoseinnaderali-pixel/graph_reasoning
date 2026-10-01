import argparse
from pathlib import Path
from gcr_llm.config import ExperimentConfig
from gcr_llm.runner import run_dataset

def main():
    p=argparse.ArgumentParser(description="Run one GCR-LLM experiment")
    p.add_argument("--input",required=True); p.add_argument("--config",required=True); p.add_argument("--output"); p.add_argument("--method")
    a=p.parse_args(); cfg=ExperimentConfig.from_yaml(a.config)
    out=a.output or f"results/raw/{Path(a.input).stem}_{a.method or cfg.ranking['method']}.json"
    rows=run_dataset(a.input,out,cfg,a.method); print(f"Saved {len(rows)} records to {out}")
if __name__=="__main__": main()
