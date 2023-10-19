
from typing import List
from src.dataset.discriminator_utils import TokenDiscriminator

def in_interactive_session():
    try:
        get_ipython
        return True
    except NameError:
        return False
    

def yield_default_and_one_off_discriminator_variations(*discrimator_lists: List[List[TokenDiscriminator]]):
    default_combination = [disc_list[0] for disc_list in discrimator_lists]
    yield default_combination
    for i, disc_list in enumerate(discrimator_lists):
        for discriminator in disc_list[1:]:
            new_combination = default_combination.copy()
            new_combination[i] = discriminator
            yield new_combination

def yield_default_discriminator_combination(*discriminator_lists: List[List[TokenDiscriminator]]):
    yield [disc_list[0] for disc_list in discriminator_lists]