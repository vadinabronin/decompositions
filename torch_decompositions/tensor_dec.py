def get_psnr(X,full_):
    """
    psnr = 20 * log(255/mse(A,B))/log(10)
    """
    mse = (((X - full_) ** 2).sum())/torch.tensor(np.prod(X.shape))
    psnr = 20 * (torch.log(torch.tensor(255.0)/mse**(1/2))/torch.log(torch.tensor(10.0)))
    return psnr

@torch.no_grad()
def cp4(X,rank,n_iter = 25,lmb = 0.1):
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
    return A,B,C,D
  
def cores_cp_full(A,B,C,D):
    full_ = einsum('ir,jr,kr,tr->ijkt',A,B,C,D).to(device = A.device)
    return full_
  
def cores_tucker_full(A,B,C,D,E):
    full_ = einsum('tyhr,it,jy,kh,lr->ijkl',A,B,C,D,E).to(device = A.device)
    return full_

@torch.no_grad()
def make_tucker(w,rank):
    device = w.device 
    t = tntorch.Tensor(w, ranks_tucker = rank,device = device)
    A = t.tucker_core()
    B = t.Us[0]
    C = t.Us[1]
    D = t.Us[2]
    E = t.Us[3]
    return A,B,C,D,E
  
def tucker_decomposition(w,rank):
    step = len(rank) - 1
    cores = []
    l = list(w.shape)
    n = len(w.shape)
    r = list(reversed(rank))
    sh = w.shape
    for d in list(reversed(sh)):
        w = torch.reshape(w,(d,-1))
        u,s,v = torch.svd(w)
        core = s[0] * u[:,:rank[step]]
        cores.append(core)
        w = v[:,:rank[step]]
        w = torch.reshape(w,r[:n - step] + l[:step])
        step -= 1
    perm = [i for i in reversed(range(len(rank)))]
    return [torch.permute(w,perm)] + list(reversed(cores))
