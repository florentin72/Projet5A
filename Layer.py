class Layer: 
     def __init__(self,nbNeurone,sortie):
         self.nbNeurone = nbNeurone 
         self.theta = None
         self.z = None 
         self.a = None
         self.sortie = sortie
         self.delta = None
         self.Delta = None