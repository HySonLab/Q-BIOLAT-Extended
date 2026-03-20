import numpy as np


def _tournament_select(rng, population, fitness, tournament_size: int = 3):
    idx = rng.integers(0, len(population), size=tournament_size)
    best = idx[np.argmax(fitness[idx])]
    return population[best].copy()


def _crossover(rng, p1, p2):
    n = len(p1)
    if n <= 1:
        return p1.copy(), p2.copy()
    point = rng.integers(1, n)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def _mutate(rng, x, mutation_rate: float):
    mask = rng.random(len(x)) < mutation_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def genetic_algorithm(
    model,
    n_bits: int,
    population_size: int = 64,
    n_generations: int = 150,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.02,
    elite_size: int = 4,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(population_size, n_bits), dtype=np.int32)

    fitness = np.array([model.energy(x) for x in population], dtype=np.float64)
    best_idx = int(np.argmax(fitness))
    best_x = population[best_idx].copy()
    best_score = float(fitness[best_idx])

    for _ in range(n_generations):
        elite_idx = np.argsort(fitness)[-elite_size:]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < population_size:
            p1 = _tournament_select(rng, population, fitness)
            p2 = _tournament_select(rng, population, fitness)

            if rng.random() < crossover_rate:
                c1, c2 = _crossover(rng, p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = _mutate(rng, c1, mutation_rate)
            c2 = _mutate(rng, c2, mutation_rate)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = np.asarray(new_population[:population_size], dtype=np.int32)
        fitness = np.array([model.energy(x) for x in population], dtype=np.float64)

        gen_best_idx = int(np.argmax(fitness))
        if float(fitness[gen_best_idx]) > best_score:
            best_score = float(fitness[gen_best_idx])
            best_x = population[gen_best_idx].copy()

    return best_x, best_score
