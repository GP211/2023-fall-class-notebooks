import copy
import numpy as np
from numba import njit
from generic_solver import Operator
from sep_python import FloatVector
class Bin2D(Operator):

    def __init__(self, mod, dat, xy):
        """
        Initialize the binning operator.
        """
        super().__init__(mod, dat)
        hyper = mod.get_hyper()
        
        ax0 = hyper.axes[0]
        n1, o1, d1 = ax0.n, ax0.o, ax0.d
        
        ax1 = hyper.axes[1]
        n2, o2, d2 = ax1.n, ax1.o, ax1.d
        
        num_points = xy.shape[1]
        self._i1 = np.zeros(num_points, dtype=np.int32)
        self._i2 = np.zeros(num_points, dtype=np.int32)
        self._sc = np.ones(num_points)
        
        for x_val, y_val, index, in zip(xy[0], xy[1],range(num_points)):
            self._i1[index] = (x_val - o1) / d1 + 0.5
            self._i2[index] = (y_val - o2) / d2 + 0.5
            
            # Check for out-of-bounds indices
            if self._i1[index] < 0 or self._i2[index] < 0 or self._i1[index] >= n1 or self._i2[index] >= n2:
                self._i1[index] = 0
                self._i2[index] = 0
                self._sc[index] = 0

    def forward(self, add, mod, dat):
        """
        Forward operation.
        """
        self.checkDomainRange(mod, dat)
        if not add:
            dat.zero()
        for d_val, i1_val, i2_val, sc_val in zip(dat, self._i1, self._i2, self._sc):
            d_val += sc_val * mod[i1_val, i2_val]

    def adjoint(self, add, mod, dat):
        """
        Adjoint operation.
        """
        self.checkDomainRange(mod, dat)
        if not add:
            mod.zero()

        for i1_val, i2_val, sc_val, d_val in zip( self._i1, self._i2, self._sc, dat):
            mod[i1_val, i2_val] += sc_val * d_val

class Linear2D(Operator):
  def __init__(self,model:FloatVector,data:FloatVector,x,y):
    if not isinstance(model,FloatVector) or not isinstance(data,FloatVector):
      raise Exception("wrong input")
    axes=model.get_hyper().axes
    if len(axes)!=2:
      raise Exception("expecting model to be 2-D")

    self._f1=(x-axes[0].o)/axes[0].d
    self._f2=(y-axes[1].o)/axes[1].d
    self._ipos2=np.int_(self._f2)
    self._ipos1=np.int_(self._f1)
    self._f1-=self._ipos1
    self._f2-=self._ipos2
    self._e1=1.-self._f1
    self._e2=1.-self._f2
    find_outside(axes[0].n,axes[1].n,self._ipos1,self._ipos2,self._f1,\
        self._f2,self._e1,self._e2)
    super().__init__(model,data)

  def forward(self,add,model,data):
    self.checkDomainRange(model,data)
    if not add:
      data.zero()
    d=data.get_nd_array()
    m=model.get_nd_array()
    forward_it(m,d,self._ipos1,self._ipos2,self._f1,self._f2,self._e1,self._e2)
    
  def adjoint(self,add,model,data):
    self.checkDomainRange(model,data)
    if not add:
      model.zero()
    m=model.get_nd_array()
    d=data.get_nd_array()
    adjoint_it(m,d,self._ipos1,self._ipos2,self._f1,self._f2,self._e1,self._e2)

  

@njit()
def forward_it(m,d,ipos1,ipos2,f1,f2,e1,e2):

  for i in range(ipos1.shape[0]):
   d[i]+=m[ipos2[i],ipos1[i]]*f1[i]*f2[i]+\
      m[ipos2[i]+1,ipos1[i]]*f1[i]*e2[i]+\
      m[ipos2[i],ipos1[i]+1]*e1[i]*f2[i]+\
      m[ipos2[i]+1,ipos1[i]+1]*e1[i]*e2[i]
@njit()
def adjoint_it(m,d,ipos1,ipos2,f1,f2,e1,e2):

  for i in range(ipos1.shape[0]):
   m[ipos2[i],ipos1[i]]+=d[i]*f1[i]*f2[i]
   m[ipos2[i]+1,ipos1[i]]+=d[i]*e2[i]*f1[i]
   m[ipos2[i],ipos1[i]+1]+=d[i]*e1[i]*f2[i]
   m[ipos2[i]+1,ipos1[i]+1]+=d[i]*e1[i]*e2[i]

@njit(parallel=True)
def find_outside(n1,n2,ip1,ip2,f1,f2,e1,e2):
  for i in range(ip1.shape[0]):
    if ip1[i] < 0 or ip2[i] <0 or ip1[i] >=n1-1 or\
      ip2[i] <0 or ip2[i] >=n2-1:
      f1[i]=0
      f2[i]=0
      e1[i]=0
      e2[i]=0
      ip1[i]=0
      ip2[i]=0
