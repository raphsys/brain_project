class MemorySupervisorV3:
    def __init__(self, num_classes, initial_replay=32):
        self.num_classes = num_classes
        self.initial_replay = initial_replay
        self.replay_budget = {i: initial_replay for i in range(num_classes)}
        self.class_variances = {i: 1.0 for i in range(num_classes)}
        self.memory_sizes = {i: 0 for i in range(num_classes)}

    def update_memory_info(self, class_label, variance, mem_size):
        self.class_variances[class_label] = variance
        self.memory_sizes[class_label] = mem_size

    def get_replay_budget(self, class_label):
        var = self.class_variances[class_label]
        mem = self.memory_sizes[class_label]
        factor = 1.0 + (var * 5.0)
        budget = int(self.initial_replay * factor / (mem + 1e-6))
        return max(5, min(64, budget))

