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

# Generation-server agent config (honesty_evals port) — SERVER_CONFIG is
# populated at runtime by honesty_evals/evals/among_us/run.py (see
# server_agent.py for the schema).
SERVER_LLM = {
    "Impostor": "ServerLLM",
    "Crewmate": "ServerLLM",
    "IMPOSTOR_LLM_CHOICES": ["local-server"],
    "CREWMATE_LLM_CHOICES": ["local-server"],
    "SERVER_CONFIG": {},  # Populated at runtime by the eval glue
}