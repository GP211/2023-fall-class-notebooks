from sep_python import FloatVector,get_sep_vector, Hypercube
from generic_solver._pyOperator import Operator
import copy
import numpy as np
from numba import njit

class Helix2cart:
    """Convert to and from cartesian and helix space"""
    def __init__(self,nd):
        """Initialize conversion
        
             nd - Data dimensions"""
        if not isinstance(nd,list):
            raise Exception("Expecting nd to be a list")
        
        self._ndim=copy.deepcopy(nd)
        self._b=[1]
        sz=1
        for n in self._ndim:
            if not isinstance(n,int):
                raise Exception("Expecting a list of ints")
            sz=sz*n
            self._b.append(sz)
            

    def toCart(self,hlx):
        cart=[0]*len(self._ndim)
        lft=hlx
        for i in range(len(self._ndim)).reverse():
            cart[i]=int(lft/self._b[i])
            lft-=cart[i]*self._b[i]
        return cart
        
    
    def toHelix(self,cart):
        """Convert from cartesian space to helix space"""
        if len(cart) != len(self._ndim):
            raise Exception("Expecting cart to be same size as data")
        hlx=0
        for i in range(len(self._ndim)):
            if not isinstance(cart[i],int):
                raise Exception("Expecting cart to be a list of ints")
            hlx+=cart[i]*self._b[i]
        return hlx
    
    

class HelixFilter(FloatVector):
    """Class for defining a filter on a helix"""
    def __init__(self,nd,**kw):
        """Option 1:
           nelem [int] Number of elements in filter
        
           Option 2:
            filt -  Filter to make a copy of
            
            Option 3:
             n - Length of a box describing the filter. First axis must be odd
        
        """
        self._nd=nd

        if "nelem" in kw:
            if not isinstance(kw["nelem"],int):
                raise Exception("Expecting nelem to be integer")
            super().__init__(Hypercube.set_with_ns([kw["nelem"]]))
        elif "filt" in kw:
            if not isintance(kw["filt"],HelixFilter):
                raise Exception("Expecting filter to be HelixFilter")
            super().__init__(kw["filt"]._hyper,arr=kw["filt"].get_nd_array())
        elif "n" in kw:
            self._lags=[]
            if not isinstance(kw["n"],list):
                raise Exception("Expecting n to be a list")
  
            if len(kw["n"]) > nd:
                raise Exception("Box dimensions larger than data")
            if len(kw["n"]) >3 :
                raise Exception("Can only handle 3-D")
            n=kw["n"]
            for i in range (len(nd),3):
                nd.append(1)
            for i in range(len(n),3):
                n.append(1)
            for i in range(3):
                if not isinstance(nd[i],int):
                    raise Exception("Expecting nd elements to be int")
                if not isinstance(n[i],int):
                    raise Exception("Expecting n elements to be int") 
                if n[i]>= nd[i]:
                    raise Exception("Expecting n to be smaller than nd")
            if int(n[0]/2)*2  == n[0]:
                raise Exception("Expecting first dimension to be odd")
            
            c2h=Helix2cart(nd)
            half=int(n[0]/2)
            c=[1]*3
            for i3 in range(n[2]):
                c[2]=i3
                for i2 in range(n[1]):
                    c[1]=i2
                    for i1 in range(-half,half+1):
                        c[0]=i1
                        l=c2h.toHelix(c)
                        if l>0:
                            lags.append(c2h.toHelix(c))
            
            hyper=Hypercube.set_with_ns([len(lags)])
            super().__init__(hyper)    
        else:
            raise Exception("Unknown initialization")
        
        self.lags=np.zeros((self.get_nd_array().shape[0]),dtype=np.int32)
        
    
    def clone(self):
        """Return a clone of the helix filter"""
        x=HelixFilter(filt=self)
        return x
    
    def cloneSpace(self):
        x=HelixFilter(filt=self)
        return x
    
    
class Helicon(Operator):
    """ Filtering with the helix"""
    def __init__(self,model,data,filt):
        """ 
            model - vector with hypercube and ndArray
            data  - vector with hypercube and ndArray
            filt  - HelixFilter
        
        """
        
        if not model.checkSame(data):
            raise Exception("Model and data must be same space")
            
        try:
            h=model.get_hyper()
            m=model.get_nd_array()
            h2=data.get_hyper()
            d=data.get_nd_array()
        except:
            raise Exception("Model must have a hypercube and numpy representation")
            
        if not h.check_same(h2):
            print(h,h2)
            raise Exception("Model and data must be same space") 
        
        if not isinstance(filt,HelixFilter):
            raise Exception("Expecting filt to be a helix filter")
        
        ns=h.get_ns()
        self._n123=h.get_n123()
        if len(ns) !=len(filt._nd):
            raise Exception("Expecting filter n to be the same as data")
        
        for i in range(len(ns)):
            if ns[i] != filt._nd[i]:
                raise Exception("Expecting filter n to be the same ss data")

        super().__init__(model,data)
        self._filt=filt
        
    def forward(self,add,model,data):
        """Forward helix filtering"""
        self.checkDomainRange(model,data)
        if not add:
            data.zero()
        m=np.ravel(model.get_nd_array())
        d=np.ravel(data.get_nd_array())
        data.scale_add(model)
        heliconFor(m,d,self._filt.lags,self._filt.get_nd_array())
        
    def adjoint(self,add,model,data):
        """Forward helix filtering"""
        self.checkDomainRange(model,data)
        if not add:
            model.zero()
        model.scale_add(data)
        m=np.ravel(model.get_nd_array())
        d=np.ravel(data.get_nd_array())
        heliconAdj(m,d,self._filt.lags,self._filt.get_nd_array())
               

@njit()
def heliconFor(model,data,lags,coefs):
    
    for i in range(model.shape[0]):
        for ilag in range(len(lags)):
            im=i-lags[ilag]
            if im>=0:
                data[i]+=model[im]*coefs[ilag]
@njit()
def heliconAdj(model,data,lags,coefs):
    for i in range(model.shape[0]):
        for ilag in range(len(lags)):
            im=i-lags[ilag]
            if im>=0:
                model[im]+=data[i]*coefs[ilag]


class Hconest(Operator):
    """ Filtering with the helix"""
    def __init__(self,filt,data,model):
        """ 
            model - vector with hypercube and ndArray
            data  - vector with hypercube and ndArray
            filt  - HelixFilter
        
        """
        
        if not model.checkSame(data):
            raise Exception("Model and data must be same space")
            
        try:
            h=model.get_hyper()
            m=model.get_nd_array()
            h2=data.get_hyperg()
            d=data.get_nd_array()
        except:
            raise Exception("Model must have a hypercube and numpy representation")
            
        if not h.check_same(h2):
            raise Exception("Model and data must be same space") 
        
        if not isinstance(filt,HelixFilter):
            raise Exception("Expecting filt to be a helix filter")
        
        ns=h.get_ns()
        if len(ns) !=len(filt._nd):
            raise Exception("Expecting filter n to be the same as data")
        
        for i in range(len(ns)):
            if ns[i] != filt._nd[i]:
                raise Exception("Expecting filter n to be the same ss data")
        self._n123=h.get_ns()

        super().__init__(filt,data)
        self._model=np.reshape(model.get_nd_array(),(self._n123,))
        
    def forward(self,add,filt,data):
        """Forward helix filtering"""
        self.checkDomainRange(filt,data)
        if not add:
            data.zero()
        d=np.ravel(data.get_nd_array())
        hconestFor(filt.get_nd_array(),d,filt.lags,self._model)
        
    def adjoint(self,add,filt,data):
        """Forward helix filtering"""
        self.checkDomainRange(filt,data)
        d=np.ravel(data.get_nd_array())
        if not add:
            filt.zero()
        hconestAdj(filt.get_nd_array(),d,filt.lags,self._model)
           

@njit()
def hconestFor(filt,data,lags,model):
    
    for i in range(model.shape[0]):
        for ilag in range(len(lags)):
            im=i-lags[ilag]
            if im>=0:
                data[i]+=model[im]*filt[ilag]
@njit()
def hconestAdj(filt,data,lags,model):
    for i in range(model.shape[0]):
        model[i]+=data[i]
        for ilag in range(len(lags)):
            im=i-lags[ilag]
            if im>=0:
                filt[ilag]+=data[i]* model[im]



class Polydiv(Operator):
    """ Inverse filtering with the helix"""
    def __init__(self,model,data,filt):
        """ 
            model - vector with hypercube and ndArray
            data  - vector with hypercube and ndArray
            filt  - HelixFilter
        
        """
        
        if not model.checkSame(data):
            raise Exception("Model and data must be same space")
            
        try:
            h=model.get_hyper()
            m=model.get_nd_array()
            h2=data.get_hyper()
            d=data.get_nd_array()
        except:
            raise Exception("Model must have a hypercube and numpy representation")
            
        if not h.check_same(h2):
            raise Exception("Model and data must be same space") 
        
        if not isinstance(filt,HelixFilter):
            raise Exception("Expecting filt to be a helix filter")
        
        ns=h.get_ns()
        self._n123=h.get_n123()

        if len(ns) !=len(filt._nd):
            raise Exception("Expecting filter n to be the same as data")
        
        for i in range(len(ns)):
            if ns[i] != filt._nd[i]:
                raise Exception("Expecting filter n to be the same ss data")

        super().__init__(model,data)
        self._filt=filt
        self._tt=model.clone()
        self._t=np.ravel(self._tt.get_nd_array())
    def forward(self,add,model,data):
        """Forward helix filtering"""
        self.checkDomainRange(model,data)

        if not add:
            data.zero()
        self._tt.zero()
        m=np.ravel(model.get_nd_array())
        d=np.ravel(data.get_nd_array())
        polydivFor(m,d,\
                   self._filt.lags,self._filt.get_nd_array(),self._t)
        data.scale_add(self._tt)
        
        
    def adjoint(self,add,model,data):
        """Forward helix filtering"""
        self.checkDomainRange(model,data)
        if not add:
            model.zero()
        self._tt.zero()
        m=np.ravel(model.get_nd_array())
        d=np.ravel(data.get_nd_array())
        polydivAdj(m,d,\
                   self._filt.lags,self._filt.get_nd_array(),self._t)
        model.scale_add(self._tt)

@njit()
def polydivFor(model,data,lags,coefs,tt):
    
    for i in range(model.shape[0]):
        tt[i]=model[i]
        for ilag in range(len(lags)):
            im=i-lags[ilag]
            if im>=0:
                tt[i]-=tt[im]*coefs[ilag]

@njit()
def polydivAdj(model,data,lags,coefs,tt):
    for i in range(model.shape[0]-1,0,-1):
        tt[i]=data[i]
        for ilag in range(len(lags)):
            im=i+lags[ilag]
            if im<model.shape[0]:
                tt[i]-=tt[im]*coefs[ilag]
       
