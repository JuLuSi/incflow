
class MPC(object):
    # setup
    def __init__(self, N=10, t0=0.0, tf=1.0, dt=0.1, optimizer, benchmark):
        # MPC options
        self.N = N
        self.t0 = t0 
        self.tf = tf 
        self.dt = dt # step size for forward simulation
        # optional: different step size for optimization

        # benchmark system: forward simulation of Boussinesq equation
        # provides: get_state(t) -> y
        #           set_control(u)
        #           step(t)
        self.benchmark = benchmark 

        # optimizer: solves open loop optimal control problem
        # provides: optimize(t, y, dt, N) -> u
        self.optimizer = optimizer
        
        self.J_cl = []
        self.J_ol = []
        # ... more data structures for MPC analysis

    # single MPC step
    def step(t):
        # get current state
        y = benchmark.get_state(t)

        # compute optimal control using optimizer
        u = optimizer.optimize(t, y, self.dt, self.N)

        # apply optimal control to the benchmark system
        benchmark.set_control(u)
        benchmark.step(t)

        # ... store results

    # run MPC loop
    def loop():
        t = self.t0
        while t <= self.tf
            self.step(t)
            
            t += self.dt
