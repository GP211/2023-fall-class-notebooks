import copy
import numpy as np
from sep_python import FloatVector, Hypercube
from generic_solver import Operator
from numba import njit
class BoxFilter(FloatVector):
    """Class for defining a filter on a box"""
    def __init__(self,sh,mask,zero_lag,vals=None,space_only=False):
        """
        Create a filter that is a n-d cube with some of the fixed (probably zeroed
        
        sh   - N-d arry of shape
        zero_lag - Location of the zero lag in the filter
        
        mask - Mask for filter coefs n-d array (should be 0 or 1
        vals - Value for filter coeficients
        
        space_only - Whether or not actually have storage
        """
        if tuple(sh)!= mask.shape:
            raise Exception(f"Mask {mask.shape} and filter {sh} shape must be the same")
        
        if vals is not None:
            if vals.shape != tuple(sh):
                raise Exception("vals not the same shape")
        sh_fortran=list(sh)
        sh_fortran.reverse()
        hyper=Hypercube.set_with_ns(sh_fortran)
        super().__init__(hyper,vals=vals)    
        self.mask=copy.deepcopy(mask)   
        self.zeroL=np.asarray(zero_lag)
        
    @classmethod
    def PEF(cls,sh,one_loc):
        if len(sh) != len(one_loc):
            raise Exception("One location is not the same size as box filter shape")
        b=1
        one_1d=0
        sh_use=list(sh)
        one_use=list(one_loc)
        sh_use.reverse()
        one_use.reverse()
        for filt_s, one_s in zip(sh_use,one_use):
            if one_s < 0 or one_s >= filt_s:
                raise Exception("Illegal locaiton for one in the filter")
            one_1d+=one_s*b
            b*=filt_s
            
        msk=np.ones(sh,dtype=np.float32)
        m=np.ravel(msk)
        vals=np.zeros(sh,dtype=np.float32)
        v=np.ravel(vals)
        v[:one_1d]=0
        m[:one_1d+1]=0
        v[one_1d]=1
        return BoxFilter(sh,msk,one_loc,vals=vals)


                     
    def clone(self):
        """Return a clone of the helix filter"""
        x=BoxFilter(self.get_nd_array().shape,self.mask,self.zeroL,vals=self.get_nd_array())
        return x
    
    def cloneSpace(self):
        x=BoxFilter(self.get_nd_array().shape,self.mask,self.zeroL,space_only=True)
        return x

    def create_mask(self,known):
        """Given known data it will output a weighting mask
            that zeros equations that have any unknwon samples
        """
        model=known.clone()
        data=model.clone()
        flt=self.clone()
        flt.set(1.)
        flt.get_nd_array()[:,:]*=self.mask
        op=convOpAdjData(model,data,flt)
        op.forward(False,model,data)
        
        val=np.sum(flt.get_nd_array())
        ar=np.where(data.get_nd_array()==val,1,0)
        msk=data.clone()
        m=msk.get_nd_array()
        m[:]=ar[:]
        return msk


class convOpAdjData(Operator):
    def __init__(self,model,data,filt):
        if not model.checkSame(data) and not isinstance(model,FloatVector):
            raise Exception("Expecting model and data to be the same shape and float vectors")
        self.filt=filt.clone()
        
        if len(filt.get_nd_array().shape) >len(model.get_nd_array().shape):
            raise Exception("Filter must be same number of dimensions or smaller than model/data")
        super().__init__(model,data)
        
        
    
    def forward(self,add,model,data):
        self.checkDomainRange(model,data)
        if not add:
            data.zero()
        m=model.get_nd_array()
        d=data.get_nd_array()
        f=self.filt.get_nd_array()
        if len(m.shape)==2:
            if len(f.shape)==2:
                forward_2_2(m,f,d,self.filt.zeroL)
            else:
                forward_2_1(m,f,d,self.filt.zeroL)
        else:
            forward_1_1(m,f,d,self.filt.zeroL)
        
        
    
    def adjoint(self,add,model,data):
        self.checkDomainRange(model,data)
        if not add:
            model.zero()
        m=model.get_nd_array()
        d=data.get_nd_array() 
        f=self.filt.get_nd_array()
        

        if len(m.shape)==2:
            if len(f.shape)==2:
                adjoint_m_2_2(m,f,d,self.filt.zeroL)
            else:
                adjoint_m_2_1(m,f,d,self.filt.zeroL)
        else:
            adjoint_m_1_1(m,f,d,self.filt.zeroL)    
class ConvOpAdjFilter(Operator):
    def __init__(self,filt,data,model):
        if not model.checkSame(data) and not isinstance(model,FloatVector):
            raise Exception("Expecting model and data to be the same shape and float vectors")
        self.model=copy.deepcopy(model)
        
        if len(filt.get_nd_array().shape) >len(model.get_nd_array().shape):
            raise Exception("Filter must be same number of dimensions or smaller than model/data")
        super().__init__(filt,data)
        
        
    
    def forward(self,add,filt,data):

        self.checkDomainRange(filt,data)
        if not add:
            data.zero()
        m=self.model.get_nd_array()
        d=data.get_nd_array()
        ftemp=filt.clone()
        f=ftemp.get_nd_array()
        f[:]=f[:]*filt.mask[:]

        if len(m.shape)==2:
            if len(f.shape)==2: 
               forward_2_2(m,f,d,filt.zeroL)
            else:
                forward_2_1(m,f,d,filt.zeroL)
        else:
            forward_1_1(m,f,d,filt.zeroL)
        
        
    def adjoint(self,add,filt,data):
        self.checkDomainRange(filt,data)
        if not add:
            filt.zero()
        m=self.model.get_nd_array()
        d=data.get_nd_array() 
        ftemp=filt.clone()
        f=ftemp.get_nd_array()
        ftemp.zero()
        if len(m.shape)==2:
            if len(f.shape)==2:
                adjoint_f_2_2(m,f,d,filt.zeroL)
            else:
                adjoint_f_2_1(m,f,d,filt.zeroL)
        else:
            adjoint_f_1_1(m,f,d,filt.zeroL)
        f[:]=f[:]*filt.mask[:]
        filt.scale_add(ftemp)
                
@njit()
def forward_2_2(m ,f,d,zero): 
    for i2 in range(f.shape[0]-zero[0]-1,m.shape[0]-zero[0]):
        for i1 in range(f.shape[1]-zero[1]-1,m.shape[1]-zero[1]):
            for if2 in range(0,f.shape[0]):
                for if1 in range(0,f.shape[1]):
                    d[i2,i1]+=f[if2,if1]*m[i2-if2+zero[0] ,i1-if1+zero[1]]

    
@njit()
def forward_2_1(m,f,d,zero):               
    for i2 in range(m.shape):
        for i1 in range(f.shape[0]-zero[0]-1,m.shape[1]-zero[0]):
                for if1 in range(0,f.shape[1]):
                    d[i2,i1]+=f[if1]*m[i2 ,i1-if1+zero[0]]
                                 
@njit()
def forward_1_1(m,f,d,zero):             
    for i1 in range(f.shape[0]-zero[0]-1,m.shape[0]-zero[0]):
            for if1 in range(0,f.shape[1]):
                d[i1]+=f[if1]*m[i1-if1+zero[0]]

@njit()
def adjoint_m_2_2(m,f,d,zero):          
    for i2 in range(f.shape[0]-zero[0]-1,m.shape[0]-zero[0]):
        for i1 in range(f.shape[1]-zero[1]-1,m.shape[1]-zero[1]):
            for if2 in range(0,f.shape[0]):
                for if1 in range(0,f.shape[1]):
                    m[i2-if2+zero[0] ,i1-if1+zero[1]]+=d[i2,i1]*f[if2,if1]

@njit()
def adjoint_m_2_1(m,f,d,zero):  
    for i2 in range(m.shape[0]):
        for i1 in range(f.shape[0]-zero[0]-1,m.shape[1]-zero[0]):
                for if1 in range(0,f.shape[0]):
                    m[i2,i1-if1+zero[1]]+=d[i2,i1]*f[if1]
  
@njit()
def adjoint_m_1_1(m,f,d,zero):
    for i1 in range(f.shape[0]-zero[0]-1,m.shape[0]-zero[0]):
            for if1 in range(0,f.shape[0]):
                m[i1-if1+zero[1]]+=d[i1]*f[if1]
                
@njit()
def adjoint_f_2_2(m,f,d,zero):    

    for i2 in range(f.shape[0]-zero[0]-1,m.shape[0]-zero[0]):
        for i1 in range(f.shape[1]-zero[1]-1,m.shape[1]-zero[1]):
            for if2 in range(0,f.shape[0]):
                for if1 in range(0,f.shape[1]):
                    f[if2,if1]+=m[i2-if2+zero[0] ,i1-if1+zero[1]]*d[i2,i1]
@njit()
def adjoint_f_2_1(m,f,d,zero):  
    for i2 in range(m.shape[0]):
        for i1 in range(f.shape[0]-zero[0]-1,m.shape[1]-zero[0]):
                for if1 in range(0,f.shape[0]):
                    f[if1]+=m[i2,i1-if1+zero[1]]*d[i2,i1]
  
@njit()
def adjoint_f_1_1(m,f,d,zero):
    for i1 in range(f.shape[0]-zero[0]-1,m.shape[0]-zero[0]):
            for if1 in range(0,f.shape[0]):              
                f[if1]+=m[i1-if1+zero[1]]*d[i1]


