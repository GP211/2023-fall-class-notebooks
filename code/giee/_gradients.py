from generic_solver import Operator 
from sep_python import FloatVector as FloatVector
class igrad2(Operator):

  def __init__(self,mod,dat):

    if not isinstance(mod,FloatVector) or not isinstance(dat,FloatVector):
      raise Exception("model and data must be FloatVectors")
    
    nmod=mod.get_hyper().get_ns()
    ndat=dat.get_hyper().get_ns()
    if len(nmod)!=2 or len(ndat)!=3:
      raise Exception("Unacceptable dimension")
    
    if nmod[0]!=ndat[0] or nmod[1]!=ndat[1] or ndat[2]!=2:
      raise Exception("Model and data size don't work")
    
    super().__init__(mod,dat)

  def forward(self,add,model,data):
    self.checkDomainRange(model,data)
    if not add:
      data.zero()
    d=data.get_nd_array()
    m=model.get_nd_array()

    d[0,:,:-1]+=m[:,1:]-m[:,:-1]
    d[1,:-1,:]+=m[1:,:]-m[:-1,:]

  def adjoint(self,add,model,data):
    self.checkDomainRange(model,data)
    if not add:
      model.zero()

    d=data.get_nd_array()
    m=model.get_nd_array() 

    m[:,1:]+=d[0,:,:-1]
    m[:,:-1]-=d[0,:,:-1]
    m[1:,:]+=d[1,:-1,:]
    m[:-1,:]-=d[1,:-1,:]

