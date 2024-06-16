class Histogram:
  def __init__(self, name):
    self.name = name
    self.counts = {}

  def __repr__(self):
    return repr(self.counts)

  def __str__(self):
    return str(self.counts)

  def classify(self, i):
    return i

  def add(self, i):
    classified = self.classify(i)
    if classified in self.counts:
      self.counts[classified] += 1
    else:
      self.counts[classified] = 1

  def total_count(self) -> int:
    total_count = 0
    for (key, count) in self.counts.items():
      assert isinstance(key, int)
      total_count += key * count
    return total_count

  def average_count(self) -> float:
    total_count = 0
    num_items = 0
    for (key, count) in self.counts.items():
      assert isinstance(key, int)
      num_items += count
      total_count += key * count
    return total_count / num_items

  def print(self):
    print(f"{self.name} Historgram")
    for (key, count) in sorted(self.counts.items(), key=lambda x: x[1]):
      print(f"{key}: {count}")

  def to_json(self):
    return self.counts
