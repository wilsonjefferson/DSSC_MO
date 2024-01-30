from src.larp import LARP
from src.utils.utils_waterflow.cloud import CLOUD


def clouds_generator(n_clouds:int, larp:LARP, max_pop:int) -> CLOUD:
    '''
        Python generator to generate clouds, up to n_clouds.

        Arguments
        ---------
        n_clouds:int
        Integer number of clouds to generate

        larp:LARP
        An instance of the LARP model

        max_pop:int
        Integer number of drop-of-waters (dows) to generate per each cloud  

        Return
        ------
        CLOUD
        A new cloud instance      
    '''

    for _ in range(n_clouds):
        yield CLOUD(larp, max_pop)
