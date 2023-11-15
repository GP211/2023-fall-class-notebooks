import copy
from generic_solver._pyOperator import Operator
from sep_python import FloatVector

from numba import njit
class Lap2D(Operator):

    def __init__(self, mod, dat):
        """
        Laplacian operator
        """
        super().__init__(mod, dat)

        self._jop=dat.clone()
        
        

    def forward(self, add, mod, dat):
        """
        Forward operation.
        """
        self.checkDomainRange(mod, dat)
        if not add:
            dat.zero()
        lap_forward(mod.get_nd_array(),dat.get_nd_array())


    def adjoint(self, add, mod, dat):
        """
        Adjoint operation.
        """
        self.checkDomainRange(mod, dat)
        if not add:
            mod.zero()
        lap_adjoint(mod.get_nd_array(),dat.get_nd_array())

@njit
def lap_forward(mod,dat):
    for i2 in range(1,dat.shape[0]-1):
        for i1 in range(1,dat.shape[1]-1):
            dat[i2,i1]+=mod[i2,i1]*4-mod[i2-1,i1]-mod[i2+1,i1]-mod[i2,i1-1]-mod[i2,i1+1]
        
        
@njit
def lap_adjoint(mod,dat):
    for i2 in range(1,dat.shape[0]-1):
        for i1 in range(1,dat.shape[1]-1):
            mod[i2,i1]+=dat[i2,i1]*4
            mod[i2,i1-1]-=dat[i2,i1]
            mod[i2,i1+1]-=dat[i2,i1]
            mod[i2+1,i1]-=dat[i2,i1]
            mod[i2-1,i1]-=dat[i2,i1]

     

