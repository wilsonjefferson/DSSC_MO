from src.larp import LARP
from src.utils.utils_waterflow.cloud import CLOUD


def clouds_generator(n_clouds:int, larp:LARP, max_pop:int) -> CLOUD:
    for _ in range(n_clouds):
        yield CLOUD(larp, max_pop)
