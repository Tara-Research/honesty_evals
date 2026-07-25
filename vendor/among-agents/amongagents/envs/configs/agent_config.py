IMPOSTOR_LLM = {"Impostor": "LLM", "Crewmate": "Random"}

CREWMATE_LLM = {"Impostor": "Random", "Crewmate": "LLM"}

ALL_RANDOM = {"Impostor": "Random", "Crewmate": "Random"}

ALL_LLM = {
    "Impostor": "LLM", 
    "Crewmate": "LLM",
    
    # "IMPOSTOR_LLM_CHOICES": ["meta-llama/llama-3.3-70b-instruct"],
    "CREWMATE_LLM_CHOICES": ["microsoft/phi-4"],
    
    # "CREWMATE_LLM_CHOICES": ["meta-llama/llama-3.3-70b-instruct"],
    "IMPOSTOR_LLM_CHOICES": ["microsoft/phi-4"],

    }

# Steering-server agent config (honesty_evals port) — SERVER_CONFIG is
# populated at runtime by honesty_evals/evals/among_us/run.py (see
# steered_agent.py for the schema).
STEERED_LLM = {
    "Impostor": "SteeredLLM",
    "Crewmate": "SteeredLLM",
    "IMPOSTOR_LLM_CHOICES": ["local-steered"],
    "CREWMATE_LLM_CHOICES": ["local-steered"],
    "SERVER_CONFIG": {},  # Populated at runtime by the eval glue
}