from mlp import MLP

class PopulationManager():

  def __init__(self, pop_size, mlp_dims):

      self.population = []
      for i in range(pop_size):
          self.population.append(MLP(mlp_dims))
