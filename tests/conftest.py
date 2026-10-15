import torch

# The tests use at most four CPU threads, as a CI runner has; more only adds contention on a busy machine.
torch.set_num_threads(min(4, torch.get_num_threads()))
