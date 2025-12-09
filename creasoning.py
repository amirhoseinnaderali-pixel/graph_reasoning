from google import genai
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import chat_completion
from typing import Dict, Optional, Tuple, Any, List

PLANNING_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-06-05",
    "gemini-pro-latest",
    "deepseek-v3.1:671b-cloud",
    "gpt-oss:120b-cloud",
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.0-pro-exp",
    "gemini-2.0-pro-exp-02-05",
    "gemini-exp-1206",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-exp",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite-preview",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-2.5-computer-use-preview-11-11",
]

# ============== CHAIN OF THOUGHT STRUCTURE ==============
EXPECTED_COT_KEYS = {
    "thought_id",
    "timestamp",
    "thought_type",
    "content",
    "dependencies",
    "confidence",
    "reasoning_path"
}

COT_THOUGHT_TYPES = {
    "problem_analysis",
    "hypothesis",
    "exploration",
    "technique_selection",
    "lemma_formulation",
    "verification",
    "obstacle_identification",
    "synthesis",
    "conclusion"
}

# ============== MAIN OUTPUT STRUCTURE WITH COT ==============
EXPECTED_OUTPUT_KEYS = {
    "problem_restatement",
    "known_results",
    "key_obstacles",
    "problem_type",
    "proposed_approaches",
    "chosen_approach",
    "detailed_strategy",
    "critical_analysis",
    "research_roadmap",
    "confidence_level",
    "chain_of_thought"  # NEW: CoT data
}


def parse_cot_output(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse creative mathematical approach with Chain of Thought
    """
    if not raw_text:
        return None, "Empty response"
    
    candidate = raw_text.strip()
    candidate = candidate.replace("```json", "").replace("```", "").strip()
    
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None, "Response is not valid JSON"
        
        snippet = match.group(0)
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            return None, f"JSON decode error: {exc}"
    
    if not isinstance(parsed, dict):
        return None, "Response is not a JSON object"
    
    # بررسی کلیدهای الزامی
    required_keys = {
        "problem_restatement", "known_results", "key_obstacles",
        "problem_type", "proposed_approaches", "chosen_approach",
        "detailed_strategy", "critical_analysis", "research_roadmap",
        "confidence_level", "chain_of_thought"
    }
    
    missing_keys = required_keys - parsed.keys()
    if missing_keys:
        return None, f"Missing required keys: {', '.join(sorted(missing_keys))}"
    
    try:
        normalized = {
            "problem_restatement": str(parsed["problem_restatement"]).strip(),
            "known_results": str(parsed["known_results"]).strip(),
            "key_obstacles": str(parsed["key_obstacles"]).strip(),
            "problem_type": validate_problem_type(parsed["problem_type"]),
            "proposed_approaches": validate_approaches(parsed["proposed_approaches"]),
            "chosen_approach": int(parsed["chosen_approach"]),
            "detailed_strategy": validate_detailed_strategy(parsed["detailed_strategy"]),
            "critical_analysis": validate_critical_analysis(parsed["critical_analysis"]),
            "research_roadmap": validate_research_roadmap(parsed["research_roadmap"]),
            "confidence_level": validate_creative_confidence(parsed["confidence_level"]),
            "chain_of_thought": validate_chain_of_thought(parsed["chain_of_thought"])
        }
        
        if "inspiration_sources" in parsed:
            normalized["inspiration_sources"] = str(parsed["inspiration_sources"]).strip()
        if "related_problems" in parsed:
            normalized["related_problems"] = str(parsed["related_problems"]).strip()
            
    except (KeyError, ValueError, TypeError) as exc:
        return None, f"Validation error: {exc}"
    
    return normalized, None


def validate_chain_of_thought(cot_data) -> List[Dict]:
    """
    اعتبارسنجی Chain of Thought - برای تبدیل به گراف
    """
    if not isinstance(cot_data, list):
        raise ValueError("chain_of_thought must be a list of thought nodes")
    
    if len(cot_data) < 1:
        raise ValueError("At least one thought node must be present")
    
    validated_thoughts = []
    
    for i, thought in enumerate(cot_data):
        if not isinstance(thought, dict):
            raise ValueError(f"Thought {i} must be a dictionary")
        
        required_keys = {"thought_id", "thought_type", "content", "dependencies", "confidence"}
        missing = required_keys - thought.keys()
        if missing:
            raise ValueError(f"Thought {i} missing keys: {missing}")
        
        # اعتبارسنجی thought_type
        thought_type = str(thought["thought_type"]).strip().lower()
        if thought_type not in COT_THOUGHT_TYPES:
            raise ValueError(
                f"Thought {i} has invalid type '{thought_type}'. "
                f"Must be one of: {COT_THOUGHT_TYPES}"
            )
        
        # اعتبارسنجی dependencies (برای ساخت گراف)
        if not isinstance(thought["dependencies"], list):
            raise ValueError(f"Thought {i} dependencies must be a list")
        
        # اعتبارسنجی confidence
        conf = float(thought["confidence"])
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"Thought {i} confidence must be between 0.0 and 1.0")
        
        validated_thought = {
            "thought_id": str(thought["thought_id"]).strip(),
            "thought_type": thought_type,
            "content": str(thought["content"]).strip(),
            "dependencies": [str(dep).strip() for dep in thought["dependencies"]],
            "confidence": conf,
            "reasoning_path": str(thought.get("reasoning_path", "")).strip()
        }
        
        validated_thoughts.append(validated_thought)
    
    return validated_thoughts


def validate_problem_type(ptype: str) -> str:
    valid_types = {
        "algebra", "number_theory", "topology", "analysis", 
        "combinatorics", "geometry", "logic", "other"
    }
    pt = str(ptype).strip().lower()
    
    if pt not in valid_types:
        raise ValueError(f"problem_type must be one of {valid_types}, got '{pt}'")
    
    return pt


def validate_approaches(approaches) -> list[Dict]:
    if not isinstance(approaches, list):
        raise ValueError("proposed_approaches must be a list")
    
    if len(approaches) < 1:
        raise ValueError("At least one approach must be proposed")
    
    validated = []
    for i, app in enumerate(approaches):
        if not isinstance(app, dict):
            raise ValueError(f"Approach {i} must be a dictionary")
        
        required_keys = {"approach_number", "name", "core_idea", "mathematical_tools", 
                        "novelty", "required_assumptions"}
        missing = required_keys - app.keys()
        if missing:
            raise ValueError(f"Approach {i} missing keys: {missing}")
        
        validated.append({
            "approach_number": int(app["approach_number"]),
            "name": str(app["name"]).strip(),
            "core_idea": str(app["core_idea"]).strip(),
            "mathematical_tools": str(app["mathematical_tools"]).strip(),
            "novelty": str(app["novelty"]).strip(),
            "required_assumptions": str(app["required_assumptions"]).strip(),
        })
    
    return validated


def validate_detailed_strategy(strategy) -> Dict:
    if not isinstance(strategy, dict):
        raise ValueError("detailed_strategy must be a dictionary")
    
    required_keys = {"overview", "key_steps", "required_lemmas", "computational_experiments"}
    missing = required_keys - strategy.keys()
    if missing:
        raise ValueError(f"detailed_strategy missing keys: {missing}")
    
    if not isinstance(strategy["key_steps"], list) or len(strategy["key_steps"]) < 1:
        raise ValueError("key_steps must be a non-empty list")
    
    validated_steps = []
    for i, step in enumerate(strategy["key_steps"]):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i} must be a dictionary")
        
        step_keys = {"step_number", "description", "justification", "potential_techniques"}
        missing_step = step_keys - step.keys()
        if missing_step:
            raise ValueError(f"Step {i} missing keys: {missing_step}")
        
        validated_steps.append({
            "step_number": int(step["step_number"]),
            "description": str(step["description"]).strip(),
            "justification": str(step["justification"]).strip(),
            "potential_techniques": str(step["potential_techniques"]).strip(),
        })
    
    if not isinstance(strategy["required_lemmas"], list):
        raise ValueError("required_lemmas must be a list")
    
    return {
        "overview": str(strategy["overview"]).strip(),
        "key_steps": validated_steps,
        "required_lemmas": [str(lemma).strip() for lemma in strategy["required_lemmas"]],
        "computational_experiments": str(strategy["computational_experiments"]).strip(),
    }


def validate_critical_analysis(analysis) -> Dict:
    if not isinstance(analysis, dict):
        raise ValueError("critical_analysis must be a dictionary")
    
    required_keys = {"weaknesses", "assumptions", "failure_modes", "disproof_conditions"}
    missing = required_keys - analysis.keys()
    if missing:
        raise ValueError(f"critical_analysis missing keys: {missing}")
    
    return {
        "weaknesses": str(analysis["weaknesses"]).strip(),
        "assumptions": str(analysis["assumptions"]).strip(),
        "failure_modes": str(analysis["failure_modes"]).strip(),
        "disproof_conditions": str(analysis["disproof_conditions"]).strip(),
    }


def validate_research_roadmap(roadmap) -> Dict:
    if not isinstance(roadmap, dict):
        raise ValueError("research_roadmap must be a dictionary")
    
    required_keys = {"immediate_next_step", "required_collaborations", 
                    "prerequisite_problems", "timeline_estimate"}
    missing = required_keys - roadmap.keys()
    if missing:
        raise ValueError(f"research_roadmap missing keys: {missing}")
    
    return {
        "immediate_next_step": str(roadmap["immediate_next_step"]).strip(),
        "required_collaborations": str(roadmap["required_collaborations"]).strip(),
        "prerequisite_problems": str(roadmap["prerequisite_problems"]).strip(),
        "timeline_estimate": str(roadmap["timeline_estimate"]).strip(),
    }


def validate_creative_confidence(confidence: str) -> str:
    valid_levels = {"very_low", "low", "medium"}
    conf = str(confidence).strip().lower()
    
    if conf not in valid_levels:
        raise ValueError(
            f"confidence_level for unsolved problems must be one of {valid_levels}, got '{conf}'. "
            "Note: 'high' is not allowed for speculative approaches."
        )
    
    return conf


def call_google_model(model_name: str, prompt: str, api_key: str, timeout: int = 60):
    import threading
    
    client = genai.Client(api_key=api_key)
    result_container = {"response": None, "error": None, "done": False}
    
    def api_call():
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            result_container["response"] = response
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            result_container["done"] = True
    
    thread = threading.Thread(target=api_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if not result_container["done"]:
        return {
            "model": model_name,
            "output": "",
            "success": False,
            "error": f"Google API call timeout after {timeout}s"
        }
    
    if result_container["error"]:
        return {
            "model": model_name,
            "output": "",
            "success": False,
            "error": result_container["error"]
        }
    
    try:
        parsed_output, error = parse_cot_output(result_container["response"].text)
        if parsed_output is None:
            return {
                "model": model_name,
                "output": result_container["response"].text,
                "success": False,
                "error": error
            }

        return {
            "model": model_name,
            "output": parsed_output,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "model": model_name,
            "output": "",
            "success": False,
            "error": str(e)
        }


def call_ollama_model(model: str, prompt: str, api_key: str):
    try:
        output = chat_completion(model=model, prompt=prompt, api_key=api_key)
        parsed_output, error = parse_cot_output(output)
        if parsed_output is None:
            return {
                "model": model,
                "output": output,
                "success": False,
                "error": error
            }

        return {
            "model": model,
            "output": parsed_output,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "model": model,
            "output": "",
            "success": False,
            "error": str(e)
        }


def save_to_file(results: list, filename: str, output_dir: str):
    successful_results = []
    for result in results:
        if not result["success"]:
            print(f"❌ Error calling {result['model']}: {result['error']}")
            continue
        
        structured_entry = {
            "model": result["model"],
            **result["output"],
        }
        successful_results.append(structured_entry)

    if not successful_results:
        print("⚠️  No successful results to save.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(successful_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(successful_results)}/{len(results)} results to {filename}")


def call_models_parallel(prompt: str, api_key_google: str, api_key_ollama: str):
    import time
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    
    results = [None] * len(PLANNING_MODELS)
    API_TIMEOUT = 60

    def _dispatch(model: str, index: int):
        time.sleep(index * 0.3)
        try:
            if model.startswith("gemini-"):
                print(f"🔄 Calling Google model: {model}")
                return index, call_google_model(model, prompt, api_key_google)
            else:
                print(f"🔄 Calling Ollama model: {model}")
                return index, call_ollama_model(model=model, prompt=prompt, api_key=api_key_ollama)
        except Exception as e:
            print(f"❌ Error in thread for {model}: {str(e)}")
            return index, {
                "model": model,
                "output": "",
                "success": False,
                "error": f"Thread exception: {str(e)}"
            }

    max_workers = min(4, len(PLANNING_MODELS))
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_dispatch, model, idx): (idx, model) 
                  for idx, model in enumerate(PLANNING_MODELS)}
        
        for future in as_completed(futures, timeout=API_TIMEOUT * len(PLANNING_MODELS)):
            try:
                index, result = future.result(timeout=API_TIMEOUT)
                results[index] = result
            except FuturesTimeoutError:
                idx, model = futures[future]
                print(f"⏱️ Timeout for model {model} (index {idx})")
                results[idx] = {
                    "model": model,
                    "output": "",
                    "success": False,
                    "error": f"API call timeout after {API_TIMEOUT}s"
                }
            except Exception as e:
                idx, model = futures[future]
                print(f"❌ Exception for model {model}: {str(e)}")
                results[idx] = {
                    "model": model,
                    "output": "",
                    "success": False,
                    "error": f"Exception: {str(e)}"
                }

    return [result for result in results if result is not None]


def creasoning(problem: str, api_key_google: str, api_key_ollama: str, 
               filename: str, other_model_thinking: list, output_dir: str):
    
    planning_template = """
# Creative Mathematical Approach Generator with Chain of Thought

You are an expert mathematician tasked with proposing novel approaches to unsolved mathematical problems. 

**CRITICAL: You MUST document your reasoning process as a Chain of Thought (CoT) that can be converted into a directed graph.**

## Chain of Thought Structure

Your CoT must be a sequence of "thought nodes" where each node represents one reasoning step. Each thought has:
- **thought_id**: Unique identifier (e.g., "T1", "T2", "T3")
- **thought_type**: One of: problem_analysis, hypothesis, exploration, technique_selection, lemma_formulation, verification, obstacle_identification, synthesis, conclusion
- **content**: The actual reasoning/insight at this step
- **dependencies**: List of thought_ids this depends on (for graph edges)
- **confidence**: Float 0.0-1.0 indicating confidence in this reasoning step
- **reasoning_path**: Brief explanation of how you arrived at this thought

### Example CoT Flow:
```
T1 (problem_analysis) → T2 (hypothesis) → T3 (exploration)
                      ↘ T4 (obstacle_identification)
T3 → T5 (technique_selection) → T6 (synthesis) → T7 (conclusion)
```

## Output Format (MUST BE VALID JSON)

```json
{{
    "problem_restatement": "Clear statement of the unsolved problem",
    "known_results": "Summary of partial results and failed approaches",
    "key_obstacles": "Why this problem is hard",
    "problem_type": "One of: algebra, number_theory, topology, analysis, combinatorics, geometry, logic, other",
    
    "proposed_approaches": [
        {{
            "approach_number": 1,
            "name": "Brief name",
            "core_idea": "Key insight",
            "mathematical_tools": "Tools/techniques needed",
            "novelty": "How this differs from previous attempts",
            "required_assumptions": "What needs to be true"
        }}
    ],
    
    "chosen_approach": 1,
    
    "detailed_strategy": {{
        "overview": "Detailed explanation",
        "key_steps": [
            {{
                "step_number": 1,
                "description": "First major step",
                "justification": "Why this is sound",
                "potential_techniques": "Specific methods to try"
            }}
        ],
        "required_lemmas": ["Lemma 1: ...", "Lemma 2: ..."],
        "computational_experiments": "Numerical tests to try"
    }},
    
    "critical_analysis": {{
        "weaknesses": "Where this might fail",
        "assumptions": "Unproven assumptions",
        "failure_modes": "Specific breakdown scenarios",
        "disproof_conditions": "What would show this is wrong"
    }},
    
    "research_roadmap": {{
        "immediate_next_step": "First concrete action",
        "required_collaborations": "Expertise needed",
        "prerequisite_problems": "Smaller problems to solve first",
        "timeline_estimate": "Realistic timeframe"
    }},
    
    "confidence_level": "One of: very_low, low, medium",
    
    "chain_of_thought": [
        {{
            "thought_id": "T1",
            "thought_type": "problem_analysis",
            "content": "First, I analyze the problem structure and identify it as...",
            "dependencies": [],
            "confidence": 0.9,
            "reasoning_path": "Started with problem statement, examined known results"
        }},
        {{
            "thought_id": "T2",
            "thought_type": "hypothesis",
            "content": "Based on T1, I hypothesize that...",
            "dependencies": ["T1"],
            "confidence": 0.7,
            "reasoning_path": "Building on the structural analysis, I see a pattern similar to..."
        }},
        {{
            "thought_id": "T3",
            "thought_type": "exploration",
            "content": "Exploring the hypothesis from T2, I consider using...",
            "dependencies": ["T2"],
            "confidence": 0.6,
            "reasoning_path": "Testing the hypothesis against known partial results"
        }},
        {{
            "thought_id": "T4",
            "thought_type": "obstacle_identification",
            "content": "However, from T1 I notice a major obstacle: ...",
            "dependencies": ["T1"],
            "confidence": 0.85,
            "reasoning_path": "Reviewing why previous approaches failed"
        }},
        {{
            "thought_id": "T5",
            "thought_type": "technique_selection",
            "content": "To overcome T4 and advance T3, I propose using...",
            "dependencies": ["T3", "T4"],
            "confidence": 0.65,
            "reasoning_path": "Combining insights from exploration and obstacle analysis"
        }},
        {{
            "thought_id": "T6",
            "thought_type": "synthesis",
            "content": "Synthesizing T2, T3, T5, the overall approach is...",
            "dependencies": ["T2", "T3", "T5"],
            "confidence": 0.6,
            "reasoning_path": "Integrating all previous insights into coherent strategy"
        }},
        {{
            "thought_id": "T7",
            "thought_type": "conclusion",
            "content": "Final assessment: This approach is promising because... but has risks...",
            "dependencies": ["T6"],
            "confidence": 0.55,
            "reasoning_path": "Critical evaluation of the complete strategy"
        }}
    ]
}}
```

## Guidelines for Chain of Thought

1. **Start with problem_analysis** - Always begin by analyzing the problem
2. **Show your reasoning progression** - Each thought should logically flow from previous ones
3. **Branch when appropriate** - If you consider multiple paths, show them as parallel thoughts
4. **Identify obstacles explicitly** - Use obstacle_identification nodes to show critical thinking
5. **Synthesis before conclusion** - Combine insights before final assessment
6. **Be honest about confidence** - Lower confidence for more speculative steps
7. **Make dependencies clear** - This creates the graph structure

## Problem to Solve:
{problem}

## Other Models' Thinking:
{other_model_thinking}

**Remember: Your chain_of_thought is crucial for graph visualization and analysis. Make it detailed and well-structured!**
"""
    
    planning_prompt = planning_template.format(
        problem=problem,
        other_model_thinking=other_model_thinking
    )
    
    print(f"\n{'='*60}")
    print("🚀 Starting Planning Phase with Chain of Thought")
    print(f"{'='*60}\n")
    
    results = call_models_parallel(planning_prompt, api_key_google, api_key_ollama)
    
    save_to_file(results, filename, output_dir)
    
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"✅ Planning Complete: {successful}/{len(results)} models succeeded")
    print(f"{'='*60}\n")
    
    return results