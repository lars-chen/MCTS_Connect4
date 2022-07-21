import cProfile
from agents.agent_mcts import generate_move
from main import human_vs_agent

cProfile.run(
"human_vs_agent(generate_move, generate_move)", "mmab_njit_connected_four.prof"
)