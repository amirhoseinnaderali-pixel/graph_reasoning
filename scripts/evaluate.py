import argparse,json
from pathlib import Path
from gcr_llm.metrics import summarize_results

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",required=True); a=p.parse_args(); rows=[]
    for f in sorted(Path(a.results).glob("*.json")):
        data=json.loads(f.read_text(encoding="utf-8")); rows.extend(data if isinstance(data,list) else [data])
    summary=summarize_results(rows); print(json.dumps(summary,indent=2))
    if summary["accuracy"] is None: print("Accuracy: Not yet evaluated")
if __name__=="__main__": main()
