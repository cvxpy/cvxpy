import numpy as np
import torch

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestGenTorchExp(BaseTest):
    """ Unit tests for gen_torch_exp"""

    def setUp(self) -> None:
        #Tests the functionality of gen_torch_exp
        self.n = 3
        self.m = 2
        self.x = cp.Variable(self.n)
        self.w = cp.Parameter(self.n)
        self.w.value=np.ones(self.n)
        self.Q = np.array([[2,2,1],[1,-1,2],[-1,-1,1]]) #3x3
        self.a = 3*np.ones(self.n)
        self.t1 = np.random.randn(self.n)
        self.t2 = np.random.randn(self.n)
        self.T1 = torch.ones((self.m,self.n), dtype=torch.float64) #2x3
        self.T2 = torch.ones((self.m,self.n), dtype=torch.float64) #2x3
        self.X = cp.Variable((self.m,self.n))
        self.Y = cp.Parameter((self.m,self.n))
        self.c = cp.Constant(self.n)

    def test_gen_torch_exp(self):
            exp1  = self.x+self.w+self.a+self.x+self.w
            exp2  = self.x+self.w+self.a+self.x@self.w+self.x
            exp3  = cp.norm(self.Q@self.x+self.w+self.a)
            exp4  = self.x-self.w
            exp5  = self.w-self.x
            exp6  = self.X@self.Y.T
            exp7  = self.x@(self.w+self.w+self.w)
            exp8  = self.x
            exp9  = self.w
            exp10 = self.c

            torch_exp1,  _ = exp1.gen_torch_exp()
            torch_exp2,  _ = exp2.gen_torch_exp()
            torch_exp3,  _ = exp3.gen_torch_exp()
            torch_exp4,  _ = exp4.gen_torch_exp()
            torch_exp5,  _ = exp5.gen_torch_exp()
            torch_exp6,  _ = exp6.gen_torch_exp()
            torch_exp7,  _ = exp7.gen_torch_exp()
            torch_exp8,  _ = exp8.gen_torch_exp()
            torch_exp9,  _ = exp9.gen_torch_exp()
            torch_exp10, _ = exp10.gen_torch_exp()

            test1  = torch_exp1(5*torch.ones(self.n, dtype=torch.float64),
                               torch.tensor([1.,2.,3.], dtype=torch.float64))
            test2  = torch_exp2(1*torch.ones(self.n, dtype=torch.float64),
                               torch.tensor([1.,2.,3.], dtype=torch.float64))
            test3  = torch_exp3(2*torch.ones(self.n, dtype=torch.float64),
                               torch.tensor([2.,1.,2.], dtype=torch.float64))
            test4  = torch_exp4(self.t1, self.t2)
            test5  = torch_exp5(self.t1, self.t2)
            test6  = torch_exp6(self.T1, self.T2)
            test7  = torch_exp7(torch.tensor(self.t1), torch.tensor(self.t2))
            test8  = torch_exp8(self.t1)
            test9  = torch_exp9(self.t1)
            test10 = torch_exp10()

            self.assertTrue(all(test1==torch.tensor([15., 17., 19.])))
            self.assertTrue(all(test2==torch.tensor([12, 13, 14])))
            self.assertTrue(np.isclose(test3, 17.2626))
            #Variables and parameters are treated similarly
            self.assertTrue(all(np.isclose(test4, test5))) 
            self.assertTrue((test6==self.n*torch.ones((self.m,self.m))).all())
            self.assertTrue(torch.all(test7==torch.tensor(self.t1)@(3*torch.tensor(self.t2))).item())
            self.assertTrue(np.all(self.t1==test8))
            self.assertTrue(np.all(self.t1==test9))
            self.assertTrue(torch.all(test10==self.n).item())


    # def test_gen_torch_exp(self):
    #         #Tests the functionality of gen_torch_exp
    #         n = 3
    #         m = 2
    #         x = cp.Variable(n)
    #         w = cp.Parameter(n)
    #         w.value=np.ones(n)
    #         Q = np.array([[2,2,1],[1,-1,2],[-1,-1,1]], dtype=np.float64) #3x3
    #         a = 3*np.ones(n)
    #         t1 = np.random.randn(n)
    #         t2 = np.random.randn(n)
    #         T1 = np.ones((m,n)) #2x3
    #         T2 = np.ones((m,n)) #2x3
    #         X = cp.Variable((m,n))
    #         Y = cp.Parameter((m,n))

    #         exp1 = x+w+a+x+w
    #         exp2 = x+w+a+x@w+x
    #         exp3 = cp.norm(Q@x+w+a)
    #         exp4 = x-w
    #         exp5 = w-x
    #         exp6 = X@Y.T

    #         torch_exp1, _ = exp1.gen_torch_exp()
    #         torch_exp2, _ = exp2.gen_torch_exp()
    #         torch_exp3, _ = exp3.gen_torch_exp()
    #         torch_exp4, _ = exp4.gen_torch_exp()
    #         torch_exp5, _ = exp5.gen_torch_exp()
    #         torch_exp6, _ = exp6.gen_torch_exp()

    #         test1 = torch_exp1(5*torch.ones(n), torch.tensor([1,2,3]))
    #         test2 = torch_exp2(1*torch.ones(n), torch.tensor([1,2,3]))
    #         test3 = torch_exp3(2*torch.ones(n), torch.tensor([2.,1.,2.]))
    #         test4 = torch_exp4(t1, t2)
    #         test5 = torch_exp5(t1, t2)
    #         test6 = torch_exp6(T1, T2)

    #         self.assertTrue(all(test1==torch.tensor([15., 17., 19.])))
    #         self.assertTrue(all(test2==torch.tensor([12, 13, 14])))
    #         self.assertTrue(np.isclose(test3, 17.2626))
    #         #Variables and parameters are treated similarly
    #         self.assertTrue(all(np.isclose(test4, test5))) 
    #         self.assertTrue((test6==n*np.ones((m,m))).all())