class Node():

    def __init__(self, name, reward=0., parent=None, depth=0):

        self.name = name
        self.reward = reward
        self.parent = parent
        self.depth = depth

        self.children = {}
        self.is_root = parent is None
        self.visits = 0
        self.return_ = self.reward
        self.expected_return = 0
        self.expected_reward = self.reward
        self.touched = False

    def add(self, name, reward=0.):
        if name not in self.children:
            node = Node(name, reward=reward, parent=self, depth=self.depth + 1)
            self.children[name] = node
        else:
            self.children[name].reward += reward
            self.children[name].return_ += reward

        return self.children[name]

    def visit(self):
        self.visits += 1
        self.expected_reward = self.reward / self.visits
        return self

    def __repr__(self):
        return f"name: {self.name} is_root: {self.is_root} parent: {self.parent.name if self.parent else None} children: {list(self.children.keys())} visits: {self.visits} reward: {self.reward:.2f} expected_reward: {self.expected_reward:.2f} return: {self.return_:.2f} expected_return: {self.expected_return:.2f} depth: {self.depth}"

    def __str__(self):
        return self.__repr__()