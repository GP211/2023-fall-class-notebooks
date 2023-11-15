import copy
from generic_solver._pyOperator import Operator
from numba import njit
from sep_python import FloatVector

class  Causal(Operator):

  def __init__(self,mod:FloatVector,dat:FloatVector):

    if not mod.checkSame(dat):
        raise Exception("Model and data not same space")

    super().__init__(mod,dat)

  def forward(self,add,mod,dat):
    self.checkDomainRange(mod,dat)
    if not add:
      dat.zero()

    caus_forward(mod.get_nd_array(),dat.get_nd_array())

  def adjoint(self,add,mod,dat):
    self.checkDomainRange(mod,dat)
    if not add:
      mod.zero()
    caus_adjoint(mod.get_nd_array(),dat.get_nd_array())


import math  
@njit()
def caus_forward(mod,dat):
  t=0
  for imod in range(mod.shape[0]):
    t=t+mod[imod]
    dat[imod]+=t

@njit()
def caus_adjoint(mod,dat):
  t=0
  for imod in range(mod.shape[0]-1,-1,-1):
      t=t+dat[imod]
      mod[imod]+=t
    
class CausalBoth(Operator):
    def __init__(self,model,data):
        self._op=causalInt(model,data)
        self._tmp=model.clone()
        super().__init__(model,data)
    
    def forward(self,add,model,data):
        self._op.forward(False,model,self._tmp)
        self._op.adjoint(add,data,self._tmp)

    def adjoint(self,add,model,data):
        self._op.adjoint(False,self._tmp,data)
        self._op.forward(add,self._tmp,data)
    

