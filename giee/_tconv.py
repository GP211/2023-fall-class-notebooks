import numpy as np
from numba import njit
from generic_solver import Operator
from sep_python import FloatVector

class Tconv(Operator):

  def __init__(self,mod:FloatVector,dat:FloatVector,filt:np.ndarray):
    if not isinstance(mod,FloatVector) or not isinstance(dat,FloatVector):
      raise Exception(f"Expecting float vectors got {type(mod)} and {type(dat)}")
    nm=mod.get_hyper().get_ns()
    nd=dat.get_hyper().get_ns()

    if not isinstance(filt,np.ndarray):
      raise Exception("Expecting filter to be an n-d array")

    if len(list(filt.shape))!=1:
      raise Exception("Expecting filter to be 1-D")
    
    if len(nm) !=1 or len(nd)!=1:
      raise Exception("Expecting 1-D vectors")
    
    if nd[0]!=nm[0]+filt.shape[0]-1:
      raise Exception("Expecting size of data to be len(filt)+len(mod)-1")
    
    self._filt=np.copy(filt)

    super().__init__(mod,dat)

  def forward(self,add,mod,dat):
    self.checkDomainRange(mod,dat)
    if not add:
      dat.zero()

    tconv_forward(mod.get_nd_array(),dat.get_nd_array(),self._filt)

  def adjoint(self,add,mod,dat):
    self.checkDomainRange(mod,dat)
    if not add:
      mod.zero()

    tconv_adjoint(mod.get_nd_array(),dat.get_nd_array(),self._filt)
  
@njit()
def tconv_forward(mod,dat,filt):
  for imod in range(mod.shape[0]):
    for ifilt in range(filt.shape[0]):
      dat[imod+ifilt]+=filt[ifilt]*mod[imod]

@njit()
def tconv_adjoint(mod,dat,filt):
  for imod in range(mod.shape[0]):
    for ifilt in range(filt.shape[0]):
      mod[imod]+=filt[ifilt]*dat[imod+ifilt]

