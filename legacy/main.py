
from creasoning import creasoning
api_key_google = [
    "AIzaSyAtl_NM0nWfVMClZ_molEV6BW0IhDnhmxE",
    "AIzaSyBbo6_ZUbFVbgZZRhvaEq6FvM1pf2WnGsQ",
    "AIzaSyDJs0oTgwjITM6vFaVhRW17OiqaLGMowV0",
    "AIzaSyCWhyT8AJBgyHdr0jWy8LsEf0l5G64XLSY",
    
]
api_key_ollama = "ollama"

import json
from reasoning_graph import reasoning_graph

from graph import graph_of_thoughts

from page_rank import page_rank
from k_means import k_means




def path_to_solution_string(path_dict):
    solution_parts = []
    solution_parts.append(f"Path ID: {path_dict.get('id', 'N/A')}")
    solution_parts.append(f"Type: {path_dict.get('type', 'N/A')}")
    solution_parts.append(f"Description: {path_dict.get('description', 'N/A')}")
    solution_parts.append(f"Novelty Score: {path_dict.get('novelty_score', 'N/A')}")
    solution_parts.append(f"Confidence: {path_dict.get('confidence', 'N/A')}")
    solution_parts.append("\n--- Solution Steps ---\n")
        
    if 'nodes_detail' in path_dict:
        for idx, node in enumerate(path_dict['nodes_detail'], 1):
            node_type = node.get('type', 'unknown')
            if node_type == 'chunk':
                text = node.get('text', '')
                solution_parts.append(f"\nStep {idx} (Chunk from Solution {node.get('solution_id', 'N/A')}):")
                solution_parts.append(text)
            elif node_type == 'thought':
                content = node.get('content', '')
                thought_type = node.get('thought_type', 'unknown')
                solution_parts.append(f"\nStep {idx} (Thought - {thought_type}):")
                solution_parts.append(content)

    return "\n".join(solution_parts)



if __name__ == "__main__":




   
    
    problem = """HARD SIGNAL & SYSTEM PROBLEM (COPY-READY)

Problem:
Consider a continuous-time LTI system \mathcal{S} whose output is

y(t) = \frac{d}{dt}\big[\, x(t) * h(t) \,\big]

where the impulse response h(t) is unknown. The system satisfies the following properties:

(1) Eigenfunction Response
For every real number \omega, the exponential e^{j\omega t} is an eigenfunction of the system with eigenvalue

\lambda(\omega) = \frac{j\omega}{1 + \omega^2}.

(This means the frequency response is H(j\omega) = \lambda(\omega).)

(2) BIBO Stability
The system is BIBO stable.

(3) Causality
The impulse response h(t) is real-valued and satisfies h(t) = 0 for all t < 0.

Tasks

(A)
Find the exact impulse response h(t).

(B)
Determine whether the system is minimum-phase, maximum-phase, or mixed-phase.

(C)
Find the zero-state response to the input

x(t) = e^{-t}u(t) + 3\cos(4t)\,u(t).

(D)
Assume now the same system but with inputs supported only on [0,\infty). Is the system invertible? If yes, find a causal inverse system.

(E)
Determine whether the system can be implemented using only:
- integrators
- real multipliers
- and first-order stable causal filters

If possible, provide a block diagram; otherwise justify why it is impossible.
"""

    
    
    
    
    output_dir = "data"
    
     
  
    reasoning_1 = creasoning(problem, api_key_google[0], api_key_ollama, "creasoning_1.json",[],output_dir)
    reasoning_2 = creasoning(problem, api_key_google[1], api_key_ollama, "creasoning_2.json",reasoning_1,output_dir)

    reasoning_3 = creasoning(problem, api_key_google[0], api_key_ollama, "creasoning_3.json",reasoning_2,output_dir)
    reasoning_4 = creasoning(problem, api_key_google[3], api_key_ollama, "creasoning_4.json",reasoning_3,output_dir)
    reasoning_5 = creasoning(problem, api_key_google[3], api_key_ollama, "creasoning_5.json",reasoning_4,output_dir)

    reasoning_graph()
    graph_of_thoughts()
    page_rank()
    graph_of_thoughts()
    k_means()

    # Convert path dictionary to solution string

 
    













