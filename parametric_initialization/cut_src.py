from qibo import gates
from qibo.models import Circuit
import numpy as np
from scipy.optimize import minimize, basinhopping


class cut_model:
    def __init__(self, qubits, input_layers, cb_layers, epsilon = .01, ansatz='full') -> None:
        self._qubits = qubits
        self._epsilon = epsilon
        self._max_entries = int(self._qubits ** 3 / 5)
        self._confident_entries = int(self._qubits ** 3 / 4)
        self.ansatz = ansatz

        self.zero_state = np.zeros(2**self._qubits, dtype=np.complex128)
        self.zero_state[0] = 1

        self.input_layers = input_layers
        self.cb_layers = cb_layers

        self.circuit2 = Circuit(self._qubits)
        self.circuit2.add(self.circuit_input(end=True))
        self.params2 = 2 * np.pi * np.random.rand(len(self.circuit2.get_parameters()))

        self.circuit1 = Circuit(self._qubits)
        self.circuit1.add(self.circuit_input(end=False))
        self.params1 = 2 * np.pi * np.random.rand(len(self.circuit1.get_parameters()))

        self.circuit_reductor = Circuit(self._qubits)  
        self.circuit_reductor.add(self.circuit_cb())
        self.reductor_params = np.zeros(len(self.circuit_reductor.get_parameters()))
        for q in range(self._qubits): self.reductor_params[q] = -np.pi / 2

        self.update_params1(self.params1)

        self.measure_circuit = Circuit(self._qubits)
        self.measure_circuit.add(gates.M(*range(self._qubits)))

        self.M = int(self._confident_entries / self._epsilon**2)
        self.P = 1e-4


    def create_circuit_cb(self, layers):
        self.circuit_reductor = Circuit(self._qubits)
        self.circuit_reductor.add(self.circuit_cb(layers))

        if layers == 1: self.reductor_params = 2 * np.pi * np.random.rand(len(self.circuit_reductor.get_parameters()))
        else: self.reductor_params = np.concatenate(
            (self.reductor_params, 2 * np.pi * np.random.rand(len(self.circuit_reductor.get_parameters()) - len(self.reductor_params)), )
            )

    def create_circuit_input(self, layers):
        self.circuit1 = Circuit(self._qubits)
        self.circuit1.add(self.circuit_input(layers))

        if layers == 1: self.params1 = 2 * np.pi * np.random.rand(len(self.circuit1.get_parameters()))
        else: 
            self.params1 = np.concatenate(
            (self.params1, 2 * np.pi * np.random.rand(len(self.circuit1.get_parameters()) - len(self.params1)))
            )


    def update_params1(self, params1):
        self.params1 = params1
        self.circuit1.set_parameters(self.params1)

    
    def execute(self, params1, reductor_params):
        self.params1 = params1
        self.reductor_params = reductor_params

        self.circuit1.set_parameters(params1)
        self.circuit_reductor.set_parameters(reductor_params)
        state = self.circuit1.execute(self.initial_state.copy())
        state = self.circuit_reductor.execute(state)


        state, C, e = self.eliminate_small(state) 

        return state, C, e


    def execute_full(self, params1, params2, reductor_params):
        self.params1 = params1
        self.params2 = params2
        self.reductor_params = reductor_params

        self.circuit1.set_parameters(params1)
        self.circuit2.set_parameters(params2)
        self.circuit_reductor.set_parameters(reductor_params)

        state = self.circuit1.execute()
        state = self.circuit_reductor.execute(initial_state=state)

        state, C, e = self.eliminate_small(state) 
        
        state = self.circuit_reductor.invert().execute(initial_state=state)
        state = self.circuit2.execute(initial_state=state)
        
        return state, C, e

    
    def eliminate_small(self, state):
        outcomes1 = self.measure_circuit(initial_state = state, nshots = self.M).frequencies(binary=False)
        outcomes2 = self.measure_circuit(initial_state = state, nshots = self.M).frequencies(binary=False)
        bitstrings = []
        valid_measurements = 0
        outcomes1 = outcomes1.most_common()
        exp = 1
        for o1 in outcomes1:
            bitstrings.append(o1[0])
            try: 
                valid_measurements += outcomes2[o1[0]]
                m = self.M - valid_measurements
                if m < self.M * self._epsilon: 
                    exp = np.exp(-2 * self.M * (self._epsilon - m / self.M)**2)
                    if exp < self.P: break
            except: pass
       
        state_ = np.zeros_like(state)
        for i in bitstrings:
            state_[i] = state[i]

        return state_ / np.linalg.norm(state_), len(bitstrings), exp


    def circuit_cb(self):
        for q in range(self._qubits - 1, -1, -1):
            yield gates.RZ(q, theta=0)
            yield gates.RY(q, theta=0)
            yield gates.RZ(q, theta=0)
        for l in range(self.cb_layers):
            for q in range(1, self._qubits, 2):
                yield gates.CZ(q, (q+1) % self._qubits)
            for q in range(self._qubits - 1, -1, -1):
                yield gates.RZ(q, theta=0)
                yield gates.RY(q, theta=0)
                yield gates.RZ(q, theta=0)                    
            for q in range(0, self._qubits - 1, 2):
                yield gates.CZ(q, q+1)
            for q in range(self._qubits - 1, -1, -1):
                yield gates.RZ(q, theta=0)
                yield gates.RY(q, theta=0)
                yield gates.RZ(q, theta=0)

    
    def circuit_input(self, end=False):
        for l in range(self.input_layers):
            for q in range(self._qubits):
                yield gates.RZ(q, theta=0)
                yield gates.RY(q, theta=0)
                yield gates.RZ(q, theta=0)                    
            for q in range(0, self._qubits - 1, 2):
                yield gates.CZ(q, q+1)
            for q in range(self._qubits):
                yield gates.RZ(q, theta=0)
                yield gates.RY(q, theta=0)
                yield gates.RZ(q, theta=0)
            if self.ansatz == 'full':
                for q in range(1, self._qubits, 2):
                    yield gates.CZ(q, (q+1) % self._qubits)
            elif self.ansatz == 'pair':
                if l % 2 == 1:  
                    for q in range(1, self._qubits, 2):
                        yield gates.CZ(q, (q+1) % self._qubits)
                else:
                    for q in range(0, self._qubits - 1, 2):
                        yield gates.CZ(q, q+1)

        if end:
            for q in range(self._qubits):
                yield gates.RZ(q, theta=0)
                yield gates.RY(q, theta=0)
                yield gates.RZ(q, theta=0)   




    def cost_reductor(self, reductor_params):
        self.reductor_params = reductor_params
        state, C, e = self.execute(self.params1, reductor_params)

        cost = C - np.log(1 - e + self.P**4)

        return cost
    
    def callback_cost_reductor(self, params):
        if type(params) != np.ndarray:
            params = params.best.x
        state, C, e = self.execute(self.params1, params)

        try: 
            self.reductor_hist.append([C, e])
        except:
            self.reductor_hist = []
            self.reductor_hist.append([C, e])

    def callback_cost_reductor_initialize(self, X):
        if type(X) != np.ndarray:
            X = X.best.x
        self.reductor_params = X
        H, C, e = self.execute(self.params1, self.reductor_params)

        try: 
            self.initialize_reductor_hist.append([C, e])
        except:
            self.initialize_reductor_hist = []
            self.initialize_reductor_hist.append([C, e])



    def callback_cost_input_initialize(self, X):
        self.reductor_params = X
        H, C, e = self.execute(self.params1, self.params2, self.reductor_params)

        try: 
            self.initialize_hist.append([C, e])
        except:
            self.initialize_hist = []
            self.initialize_hist.append([C, e])
            


    def optimize_reductor(self, method = 'cma', options={'maxiter':150}, cb=None, sigma = None):
        if cb is None: cb = self.callback_cost_reductor

        self.reductor_hist = []
        cb(self.reductor_params)
        if method == 'cma':
            if sigma is None: sigma = self._epsilon

            from cma import fmin
            es = fmin(self.cost_reductor, self.reductor_params, sigma, callback = cb, options=options)
            
            res = {}
            res['x'] = es[0]
            res['fun'] = es[1]
            res['evalsopt'] = es[2]
            res['evals'] = es[3]
            res['iterations'] = es[4]
            res['xmean'] = es[5]
            res['stds'] = es[6]

        else:
            res = minimize(lambda x: self.cost_reductor(x), self.reductor_params, method=method, callback=cb, options=options)
            c = self.cost_reductor(res['x'])
        
        return res


    
    def initialize_reductor(self, target_params, max_steps, method='cma', options={'maxiter':100}):
        self.initialize_reductor_hist = []
        t = np.linspace(0, 1, max_steps)
        i_ = 0
        grad = np.zeros_like(self.reductor_params)
        for i in range(max_steps):
            C_, e = self.execute(target_params * t[i], self.reductor_params)[1:]
            
            if C_ >= self._max_entries:
                print("step", i, ":", C_)
                if i - 1 == i_: 
                    print('Unsuccessful activation')
                    break
                else: 
                    i_ = i - 1
                self.params1 = target_params * t[i_]
                prev_pars = self.reductor_params.copy()
                res = self.optimize_reductor(method=method, options = options, cb=self.callback_cost_reductor_initialize)
                yield res

        self.params1 = target_params
        res = self.optimize_reductor(method=method, options=options, cb=self.callback_cost_reductor_initialize)



    def optimize_input(self, method = 'cma', options={'maxiter':150}, cb=None):
        cb = self.callback_cost_input

        cb(self.params1)
        
        if method == 'basin-hopping':
            res = basinhopping(lambda x: self.cost_input(x), self.params1 + .0 * np.random.randn(len(self.params1)), callback=cb, **options)
            c = self.cost_input(res['x'])
            

        elif method == 'cma':
            def bounded_cost(x):
                return min(self.cost_input(x), self._c)

            from cma import fmin2
            res = fmin2(bounded_cost, self.params1, callback = cb)
            
        else:
            res = minimize(lambda x: self.cost_input(x), self.params1 + .0 * np.random.randn(len(self.params1)), method=method, callback=cb, options=options)
            c = self.cost_input(res['x'])
        
        return res


    def initialize_input(self, target_params, max_steps, method='cma', options={'maxiter':100}):
        self.initialize_input_hist = []
        t = np.linspace(0, 1, max_steps)
        for i in range(max_steps):
            self.reductor_params = target_params * t[i]
            C_, e = self.execute(self.params1, self.reductor_params)[1:]
            print(i, C_)
            
            if C_ >= self._max_entries:
                print("step", i, ":", C_)
                self.reductor_params = target_params * t[i - 1]
                res = self.optimize_input(method=method, options = options, cb=self.callback_cost_input_initialize)
                yield res

        self.reductor_params = target_params * t[i - 1]
        res = self.optimize_input(method=method, options=options, cb=self.callback_cost_input_initialize)

        yield res


class cut_qaoa(cut_model):
    def __init__(self, qubits, input_layers, cb_layers, epsilon=0.01, ham='tfim', h=.5) -> None:
        super().__init__(qubits, input_layers, cb_layers, epsilon)

        from qibo.models.variational import QAOA
        from qibo.hamiltonians import TFIM

        self.ham = TFIM(self._qubits, h=h, dense=False)
        self.h = h


        self.circuit2 = QAOA(self.ham)
        self.params2 = 2 * np.pi * np.random.rand(2 * self.input_layers)

        self.circuit1 = QAOA(self.ham)
        self.params1 = 2 * np.pi * np.random.rand(2 * self.input_layers)



class cut_tfim(cut_model):
    def __init__(self, qubits, input_layers, cb_layers, epsilon=0.01) -> None:
        super().__init__(qubits, input_layers, cb_layers, epsilon) 
        del self.ansatz

        self.circuit2 = Circuit(self._qubits)
        self.circuit2.add(self.circuit_input(end=True))
        self.params2 = 2 * np.pi * np.random.rand(len(self.circuit2.get_parameters()))

        self.circuit1 = Circuit(self._qubits)
        self.circuit1.add(self.circuit_input(end=False))
        self.params1 = 2 * np.pi * np.random.rand(len(self.circuit1.get_parameters()))

        self.update_params1(self.params1)
        


    def circuit_input(self, end=False):
        for q in range(self._qubits):
            yield gates.H(q)
        for l in range(self.input_layers):
            for q in range(0, self._qubits - 1, 2):
                yield gates.CNOT(q, q + 1)
                yield gates.RZ(q + 1, theta=0)
                yield gates.CNOT(q, q + 1)
            for q in range(1, self._qubits, 2):
                yield gates.CNOT(q, (q + 1)%self._qubits)
                yield gates.RZ((q + 1)%self._qubits, theta=0)
                yield gates.CNOT(q, (q + 1)%self._qubits)
            for q in range(self._qubits):
                yield gates.RX(q, theta=0)    


    def execute(self, params1, reductor_params):
        self.params1 = params1
        self.reductor_params = reductor_params
        self.circuit1.set_parameters(params1)

        self.circuit_reductor.set_parameters(reductor_params)

        state = self.circuit1.execute()
        
        state = self.circuit_reductor.execute(initial_state=state)
        state, C, e = self.eliminate_small(state) 

        return state, C, e
            

    def initialize_input(self, target_params, max_steps, method='cma', options={'maxiter':100}):
        self.initialize_input_hist = []
        for i in range(max_steps):
            self.t = i / max_steps
            self.reductor_params = target_params * self.t
            C_, e = self.execute(self.params1, self.reductor_params)[1:]
            
            if C_ >= self._max_entries:
                print("step", i, ":", C_)
                self.t = (i - 1) / max_steps
                self.reductor_params = target_params * self.t
                res = self.optimize_input(method=method, options = options, cb=self.callback_cost_input_initialize)
                yield res

        self.reductor_params = target_params * t[i - 1]
        res = self.optimize_input(method=method, options=options, cb=self.callback_cost_input_initialize)

        yield res


    def initialize_reductor(self, target_params, max_steps =  10, method='cma', options={'maxiter':50}, options_end={'maxfevals':np.inf, 'maxiter':500}):
        self.update_params1(target_params)
        self.initialize_reductor_hist = []
        i_ = 1
        success = True
        sigma = self._epsilon
        self.reductor_params = np.zeros_like(self.reductor_params)
        i = 1
        for q in range(self._qubits):
            self.reductor_params[3 * q + 1] = -np.pi / 2
            
        while i < len(self.params1):
            pars = np.array([target_params[_] for _ in range(i)] + [0] * (len(target_params) - i))
            self.update_params1(pars)
            self.callback_cost_reductor_initialize(self.reductor_params)
            C_, e = self.initialize_reductor_hist[-1]
            print(C_, i)
            
            if C_ > self._max_entries:
                self.initialize_reductor_hist.pop()
                pars = np.array([target_params[_] for _ in range(i - 1)] + [target_params[i - 1]] + [0] * (len(target_params) - i))
                self.update_params1(pars)
                for j in range(5):
                    res = self.optimize_reductor(method=method, options = options, cb=self.callback_cost_reductor_initialize, sigma=sigma)
                    
                    if res['iterations'] > 10: 
                        print('success 1')
                        i_ = i - 1
                        break
                    else: 
                        print('Try again with smaller sigma')
                        sigma = 0.75 * sigma

                yield res
                C_, e = self.initialize_reductor_hist[-1]
                if C_ > self._max_entries:
                    res['status'] = 'failed optimization'     
                    yield res
                    i = len(self.params1)     
                    
            if i % self._qubits == 0:
                res = self.optimize_reductor(method=method, options = options, cb=self.callback_cost_reductor_initialize, sigma=sigma)
                yield res   
            i += 1
        res = self.optimize_reductor(method=method, options = options, cb=self.callback_cost_reductor_initialize, sigma=sigma)
        yield res  
