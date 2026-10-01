import re

def normalize_answer(text):
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s.\-+/=]"," ",str(text).lower().strip())).strip()

def evaluate_selection(candidates,selected_candidate_id,gold_answer=None):
    if selected_candidate_id is None:return None
    c=next((x for x in candidates if x.candidate_id==selected_candidate_id),None)
    if c is None:return None
    if c.is_correct is not None:return bool(c.is_correct)
    if gold_answer is None:return None
    return normalize_answer(c.answer)==normalize_answer(gold_answer)

def summarize_results(records):
    valid=[r for r in records if r.get("correct") is not None]
    return {"num_records":len(records),"num_evaluated":len(valid),"accuracy":sum(bool(r["correct"]) for r in valid)/len(valid) if valid else None}
