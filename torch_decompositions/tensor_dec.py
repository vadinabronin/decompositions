import torch
import numpy as np
import tntorch
from opt_einsum_torch import einsum
import ot
def get_psnr(X,full_):
    """
    psnr = 20 * log(255/mse(A,B))/log(10)
    """
    mse = (((X - full_) ** 2).sum())/torch.tensor(np.prod(X.shape))
    psnr = 20 * (torch.log(torch.tensor(255.0)/mse**(1/2))/torch.log(torch.tensor(10.0)))
    return psnr

@torch.no_grad()
def cp4(X,rank,n_iter = 25,lmb = 0.1,return_full_matrix = True):
    m1 = X.shape[0] 
    m2 = X.shape[1] 
    m3 = X.shape[2]
    m4 = X.shape[3]      
    R = rank         
    device = X.device 
    A = torch.zeros(m1,R,device = device,dtype = torch.float32)  
    B = torch.zeros(m2,R,device = device,dtype = torch.float32)
    C = torch.zeros(m3,R,device = device,dtype = torch.float32) 
    D = torch.zeros(m4,R,device = device,dtype = torch.float32)
    torch.nn.init.xavier_uniform_(A)
    torch.nn.init.xavier_uniform_(B)
    torch.nn.init.xavier_uniform_(C)
    torch.nn.init.xavier_uniform_(D)
    for i in range(n_iter):
        f = einsum('mjkt,jl,kl,tl->lm',X,B,C,D).to(device = device)    
        g = einsum('jr,kr,tr,jl,kl,tl->lr',B,C,D,B,C,D).to(device = device)
        A = torch.transpose(torch.linalg.solve(g + lmb * torch.eye(g.shape[0]).to(device = device),f), 0, 1)
        
        f = einsum('imkt,il,kl,tl->lm',X,A,C,D).to(device = device)
        g = einsum('ir,kr,tr,il,kl,tl->lr',A,C,D,A,C,D).to(device = device)
        B = torch.transpose(torch.linalg.solve(g + lmb * torch.eye(g.shape[0]).to(device = device),f), 0, 1)
        
        f = einsum('ijmt,il,jl,tl->lm',X,A,B,D).to(device = device)
        g = einsum('ir,jr,tr,il,jl,tl->lr',A,B,D,A,B,D).to(device = device)
        C = torch.transpose(torch.linalg.solve(g + lmb * torch.eye(g.shape[0]).to(device = device),f), 0, 1)
        
        f = einsum('ijkm,il,jl,kl->lm',X,A,B,C).to(device = device)
        g = einsum('ir,jr,kr,il,jl,kl->lr',A,B,C,A,B,C).to(device = device)
        D = torch.transpose(torch.linalg.solve(g + lmb * torch.eye(g.shape[0]).to(device = device),f), 0, 1)
    if(return_full_matrix):
        full_ = einsum('ir,jr,kr,tr->ijkt',A,B,C,D).to(device = A.device)
        return A,B,C,D,full_
    else:
        return A,B,C,D
  
def cores_cp_full(A,B,C,D):
    full_ = einsum('ir,jr,kr,tr->ijkt',A,B,C,D).to(device = A.device)
    return full_
  
def cores_tucker_full(A,B,C,D,E):
    full_ = einsum('tyhr,it,jy,kh,lr->ijkl',A,B,C,D,E).to(device = A.device)
    return full_

@torch.no_grad()
def make_tucker(w,rank,return_full_matrix = True):
    device = w.device 
    t = tntorch.Tensor(w, ranks_tucker = rank,device = device)
    A = t.tucker_core()
    B = t.Us[0]
    C = t.Us[1]
    D = t.Us[2]
    E = t.Us[3]
    if(return_full_matrix):
        full_ = einsum('tyhr,it,jy,kh,lr->ijkl',A,B,C,D,E).to(device = A.device)
        return A,B,C,D,E,full_
    else:
        return A,B,C,D,E 

def svd_to_full(A,B):
    return torch.mm(A,B)

@torch.no_grad()
def make_svd(w,r,return_full_matrix = True):
    u,s,v = torch.svd(w)
    A = torch.mm(u[:,:r], torch.diag(s[:r]))
    B = v[:,:r].t()
    if(return_full_matrix):
        full_ = torch.mm(A,B)
        return A,B,full_
    else:
        return A,B
    
def get_n_perm(k,n):
    a = [i for i in range(n)]
    a[0] = k
    a[k] = 0
    return a

def findM(A,B,dim):
    g = get_n_perm(dim,len(A.shape))
    A_ = torch.permute(A,g)
    B_ = torch.permute(B,g)
    s = A_.shape[0]
    M = torch.zeros((s,s)).float().to(device = 'cuda')
    for i in range(s):
        for j in range(s):
            M[i][j] = torch.norm(A_[i] - B_[j]).cpu().item()
    return M

def findP(A,B,dim):  
    M = findM(A,B,dim)
    s = A.shape[dim]
    a = torch.ones((s,)).to(device = 'cuda')
    b = torch.ones((s,)).to(device = 'cuda')
    T = ot.emd(a, b, M)
    #T = ot.sinkhorn(a,b,M,1)
    return T

def apply_perm(A,perms):
    A_ = torch.clone(A)
    n = len(A.shape)
    for i,p in enumerate(perms):
        g = get_n_perm(i,n)
        A_ = torch.permute(A_,g)
        A_ = A_[p]
        A_ = torch.permute(A_,g)
    return A_

def basic_perm(n):
    a = [i for i in range(n)]
    return a

def findPWC(A,projection_func,dims = None,n_iter_als = 5,verbose = True,**kwargs):
    n = len(A.shape)
    perms = [torch.tensor([i for i in range(n_)])[torch.randperm(n_)] for n_ in A.shape]
    W = torch.randn(A.shape).to(device = 'cuda')
    if(dims is None):
        dims = [i for i in range(n)]
    k = 0.5/n_iter_als
    t = 0.5
    for i in range(n_iter_als):
        for d in dims:
            W = (1 - t) * W + t * projection_func(apply_perm(A,perms),**kwargs)
            perms[d] = basic_perm(A.shape[d])
            P = findP(W,apply_perm(A,perms),d)
            perm_ = torch.argmax(P, dim = 1)
            perms[d] = perm_
        err = torch.norm(apply_perm(A,perms) - W)
        if(verbose):
            print('iter number: {},err: {}'.format(i,err))
        t += k
    return perms,W

def vec(a):
    return torch.reshape(a.t(),(np.prod(a.shape),))

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def from_vec(a,shapes): 
    return reshape_fortran(a,shapes)

def kron_dec(A,shapes,num_cores = 2):
    m1,m2,n1,n2 = shapes
    #A = torch.reshape(A,(m1,n1,m2,n2))
    vecs = []
    for j in range(n1):
        for i in range(m1):
            vecs.append(vec(A[i * m2:(i + 1) * m2,j * n2:(j + 1) * n2]).cpu().numpy())
    A_ = torch.tensor(np.vstack(tuple(vecs)),device = 'cuda')
    u,s,v = torch.svd(A_)
    AB = []
    for i in range(num_cores):
        vecU0 = s[i] * u[:,i:i+1]
        vecV0 = v[:,i:i+1]
        AB.append((from_vec(vecU0,(m1,n1)),from_vec(vecV0,(m2,n2))))
    return AB

def kron_proj(A,shapes,num_cores = 2):
    AB = kron_dec(A,shapes,num_cores)
    res_ = torch.kron(AB[0][0],AB[0][1])
    for i in range(1,len(AB)):
        res_ += torch.kron(AB[i][0],AB[i][1])
    return res_
