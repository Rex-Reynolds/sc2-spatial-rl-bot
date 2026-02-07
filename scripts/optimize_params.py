#!/usr/bin/env python3
"""
Genetic algorithm to optimize bot parameters.

Uses DEAP (Distributed Evolutionary Algorithms in Python) to evolve
bot parameters by running tournaments and selecting the best performers.
"""

import argparse
import sys
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot
from sc2.data import Race, Result

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.idle_bot import IdleBot
from bots.rush_bot import RushBot
from bots.defense_bot import DefenseBot

# Try to import DEAP, provide helpful error if not installed
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("Error: DEAP library not installed.")
    print("Install with: pip install deap")
    sys.exit(1)


# Parameter ranges for each bot type
PARAM_RANGES = {
    "RushBot": {
        "MAX_WORKERS": (12, 20),
        "BARRACKS_COUNT": (1, 4),
        "ATTACK_MARINE_THRESHOLD": (4, 16),
    },
    "DefenseBot": {
        "MAX_WORKERS": (16, 24),
        "BUNKER_COUNT": (1, 3),
        "BARRACKS_COUNT": (2, 5),
        "ATTACK_MARINE_THRESHOLD": (15, 30),
    },
}


class ParameterizedBot:
    """Wrapper to create bot instances with custom parameters."""

    def __init__(self, bot_class, params: Dict):
        self.bot_class = bot_class
        self.params = params

    def __call__(self):
        """Create a new bot instance with parameters applied."""
        bot = self.bot_class()

        # Set parameters as attributes on the bot instance
        # This requires modifying the bot to read from self instead of module constants
        for key, value in self.params.items():
            setattr(bot, key.lower(), value)

        return bot


def create_bot_with_params(bot_class, params: Dict):
    """
    Create a bot instance with modified parameters.

    Note: This dynamically patches the bot's module constants.
    For production use, consider refactoring bots to accept parameters in __init__.
    """
    import importlib

    bot_module = sys.modules[bot_class.__module__]

    # Store original values
    original_values = {}
    for key, value in params.items():
        if hasattr(bot_module, key):
            original_values[key] = getattr(bot_module, key)
            setattr(bot_module, key, value)

    # Create bot instance
    bot = bot_class()

    # Restore original values
    for key, value in original_values.items():
        setattr(bot_module, key, value)

    return bot


def evaluate_parameters(
    bot_name: str,
    bot_class,
    params: Dict,
    opponent_class,
    num_matches: int = 5,
    map_name: str = "Simple64",
    time_limit: int = 300,
) -> float:
    """
    Evaluate a set of parameters by running matches against an opponent.

    Returns win rate (0.0 to 1.0).
    """
    wins = 0

    for _ in range(num_matches):
        try:
            # Dynamically set parameters in the bot's module
            bot_module = sys.modules[bot_class.__module__]
            original_values = {}

            for key, value in params.items():
                if hasattr(bot_module, key):
                    original_values[key] = getattr(bot_module, key)
                    setattr(bot_module, key, value)

            # Run game
            result = run_game(
                maps.get(map_name),
                [
                    Bot(Race.Terran, bot_class()),
                    Bot(Race.Terran, opponent_class()),
                ],
                realtime=False,
                game_time_limit=time_limit,
            )

            # Restore original values
            for key, value in original_values.items():
                setattr(bot_module, key, value)

            if result[0] == Result.Victory:
                wins += 1

        except Exception as e:
            print(f"Error in match: {e}")

    win_rate = wins / num_matches
    return win_rate


def optimize_bot(
    bot_name: str,
    bot_class,
    opponent_class,
    param_ranges: Dict,
    population_size: int = 20,
    generations: int = 10,
    matches_per_eval: int = 3,
    map_name: str = "Simple64",
) -> Tuple[Dict, float]:
    """
    Use genetic algorithm to find optimal parameters.

    Returns: (best_params, best_fitness)
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZING {bot_name}")
    print(f"{'='*70}")
    print(f"Parameter ranges: {param_ranges}")
    print(f"Population size: {population_size}")
    print(f"Generations: {generations}")
    print(f"Matches per evaluation: {matches_per_eval}")
    print()

    # Setup DEAP
    # Create fitness and individual classes
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax, params=dict)

    toolbox = base.Toolbox()

    # Parameter genes
    param_names = list(param_ranges.keys())
    for param_name in param_names:
        min_val, max_val = param_ranges[param_name]
        toolbox.register(
            f"attr_{param_name}",
            random.randint,
            min_val,
            max_val,
        )

    # Individual and population
    def create_individual():
        """Create an individual with random parameters."""
        ind = creator.Individual(
            [getattr(toolbox, f"attr_{name}")() for name in param_names]
        )
        ind.params = dict(zip(param_names, ind))
        return ind

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[param_ranges[p][0] for p in param_names], up=[param_ranges[p][1] for p in param_names], indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evaluation function
    def eval_individual(individual):
        """Evaluate an individual by running matches."""
        params = dict(zip(param_names, individual))
        individual.params = params

        win_rate = evaluate_parameters(
            bot_name,
            bot_class,
            params,
            opponent_class,
            num_matches=matches_per_eval,
            map_name=map_name,
        )
        return (win_rate,)

    toolbox.register("evaluate", eval_individual)

    # Run evolution
    population = toolbox.population(n=population_size)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum([v[0] for v in x]) / len(x))
    stats.register("max", lambda x: max([v[0] for v in x]))
    stats.register("min", lambda x: min([v[0] for v in x]))

    print("Starting evolution...")
    print()

    # Evolution loop
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")

        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Print statistics
        record = stats.compile(population)
        print(f"  Avg fitness: {record['avg']:.3f}")
        print(f"  Max fitness: {record['max']:.3f}")
        print(f"  Min fitness: {record['min']:.3f}")

        # Get best individual
        best_ind = tools.selBest(population, 1)[0]
        print(f"  Best params: {best_ind.params}")
        print()

        # Selection and breeding
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < 0.3:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Replace population
        population[:] = offspring

    # Get final best individual
    best_ind = tools.selBest(population, 1)[0]
    best_params = dict(zip(param_names, best_ind))
    best_fitness = best_ind.fitness.values[0]

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best parameters: {best_params}")
    print(f"Best win rate: {best_fitness:.3f}")
    print()

    return best_params, best_fitness


def main():
    parser = argparse.ArgumentParser(
        description="Optimize bot parameters using genetic algorithm"
    )
    parser.add_argument(
        "--bot",
        choices=["RushBot", "DefenseBot"],
        default="RushBot",
        help="Bot to optimize (default: RushBot)",
    )
    parser.add_argument(
        "--opponent",
        choices=["IdleBot", "RushBot", "DefenseBot"],
        default="IdleBot",
        help="Opponent to train against (default: IdleBot)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=20,
        help="Population size (default: 20)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations (default: 10)",
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=3,
        help="Matches per parameter evaluation (default: 3)",
    )
    parser.add_argument(
        "--map",
        default="Simple64",
        help="Map name (default: Simple64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for best parameters (JSON)",
    )

    args = parser.parse_args()

    # Select bot classes
    bots = {
        "RushBot": RushBot,
        "DefenseBot": DefenseBot,
        "IdleBot": IdleBot,
    }

    bot_class = bots[args.bot]
    opponent_class = bots[args.opponent]
    param_ranges = PARAM_RANGES[args.bot]

    # Run optimization
    best_params, best_fitness = optimize_bot(
        args.bot,
        bot_class,
        opponent_class,
        param_ranges,
        population_size=args.population,
        generations=args.generations,
        matches_per_eval=args.matches,
        map_name=args.map,
    )

    # Save results
    results = {
        "bot": args.bot,
        "opponent": args.opponent,
        "best_params": best_params,
        "best_fitness": best_fitness,
        "timestamp": datetime.now().isoformat(),
        "population_size": args.population,
        "generations": args.generations,
        "matches_per_eval": args.matches,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        # Save to default location
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{args.bot}_vs_{args.opponent}_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
