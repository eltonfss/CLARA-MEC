import numpy as np

def heuristic_offloading_policy(state, threshold=1e6):

    task_size = state[0]

    if task_size > threshold:
        decision = 1
    else:
        decision = 0

    resource = 0

    return decision, resource