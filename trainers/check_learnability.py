import os
import sys

#each param name must contain at least one friendly tag
def check_learnability(param_names, friendly_tags):
    assert(all([any([t in param_name for t in friendly_tags]) for param_name in param_names]))
