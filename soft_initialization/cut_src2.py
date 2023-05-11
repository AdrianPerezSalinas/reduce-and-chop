from pyparsing import str_type
from qibo import gates
from qibo.models import Circuit
import numpy as np
from scipy.optimize import minimize 


class cut_model2:
    def __init__(self, qubits, input_layers, cb_layers, epsilon = .01) -> None:
        self._qubits = qubits
        self._epsilon = epsilon
        self._max_entries = self._qubits ** 3 / 4
        self._confident_entries = int(self._max_entries * 1.15)
        self._min_entries = self._qubits
        
        self.input_layers = input_layers
        self.cb_layers = cb_layers

        self.circuit_reductor1 = Circuit(self._qubits)
        self.circuit_reductor1.add(self.circuit_cb())

        self.circuit_reductor2 = Circuit(self._qubits)
        self.circuit_reductor2.add(self.circuit_cb())

        self.num_reductor_params1 = len(self.circuit_reductor1.get_parameters())
        self.num_reductor_params2 = len(self.circuit_reductor2.get_parameters())

        self.reductor_params1 = np.zeros(self.num_reductor_params1)
        self.reductor_params2 = np.zeros(self.num_reductor_params2)

        self.circuit3 = Circuit(self._qubits)
        self.circuit3.add(self.circuit_input(end=True))
        self.params3 = 2 * np.pi * np.random.rand(len(self.circuit3.get_parameters()))

        self.circuit2 = Circuit(self._qubits)
        self.circuit2.add(self.circuit_input_inside(end=False))
        self.params2 = 2 * np.pi * np.random.rand(len(self.circuit2.get_parameters()))

        self.circuit1 = Circuit(self._qubits)
        self.circuit1.add(self.circuit_input(end=False))
        self.params1 = 2 * np.pi * np.random.rand(len(self.circuit1.get_parameters()))



    def update_t1(self, t1):
        self.t1 = t1
        self.state_t1 = np.cos(np.pi / 2 * self.t1) * self.zero_state + np.sin(np.pi / 2 * self.t1)*self.initial_state1

    def get_final_state1(self):
        state, C1, e1 = self.execute1(self.reductor_params1)
        self.final_state1 = state

    def update_t2(self, t2):
        self.t2 = t2

        state =  self.circuit_reductor1.invert().execute(self.final_state1.copy())
        self.initial_state2 = self.circuit2.execute(state)

        self.state_t2 = np.cos(np.pi / 2 * self.t2) * self.final_state1 + np.sin(np.pi / 2 * self.t2)*self.initial_state2


    def execute1(self, reductor_params1):
        self.reductor_params1 = reductor_params1
        self.circuit_reductor1.set_parameters(reductor_params1)
        state = self.circuit_reductor1.execute(initial_state=self.state_t1.copy()).numpy() 

        state, C1, e1 = self.eliminate_small(state) 

        return state, C1, e1


    def execute2(self, reductor_params2):
        self.reductor_params2 = reductor_params2

        self.circuit_reductor2.set_parameters(reductor_params2)
        
        state = self.circuit_reductor2.execute(initial_state=self.state_t2.copy())
        state, C2, e2 = self.eliminate_small(state)
        
        return state, C2, e2


    
    def execute_full(self, params1, params2, params3, reductor_params1, reductor_params2):
        self.params1 = params1
        self.params2 = params2
        self.params3 = params3
        self.reductor_params1 = reductor_params1
        self.reductor_params2 = reductor_params2

        self.circuit1.set_parameters(params1)
        self.circuit2.set_parameters(params2)
        self.circuit3.set_parameters(params3)
        self.circuit_reductor1.set_parameters(reductor_params1)
        self.circuit_reductor2.set_parameters(reductor_params2)

        state = self.circuit1.execute()
        state = self.circuit_reductor1.execute(initial_state=state).numpy() 

        state, C1, e1 = self.eliminate_small(state) 
        
        state = self.circuit_reductor1.invert().execute(initial_state=state)
        state = self.circuit2.execute(initial_state=state)
        state = self.circuit_reductor2.execute(initial_state=state).numpy() 

        state, C2, e2 = self.eliminate_small(state) 
        
        state = self.circuit_reductor2.invert().execute(initial_state=state)
        state = self.circuit3.execute(initial_state=state)

        return state, (C1, C2), (e1, e2)

    
    def eliminate_small(self, state):
        indices = np.argsort(np.abs(state)**2)[::-1]
        s = np.cumsum(np.abs(state[indices])**2)
        wh = np.where(s < 1 - self._epsilon)[0]

        C = len(wh) + 1
        
        state_ = np.zeros_like(state)
        for i in indices[:C]:
            state_[i] = state[i]

        return state_ / np.linalg.norm(state_), C, 1 - s[C]


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

    def cost_reductor1(self, reductor_params1):
        self.reductor_params1 = reductor_params1
        state, C, e = self.execute1(reductor_params1)

        cost = C

        return cost

    
    def cost_reductor2(self, reductor_params2):
        state, C, e = self.execute2(reductor_params2)

        cost = C

        return cost


    def optimize_reductor1(self, method = 'cma', options={'maxiter':150}, cb=None, gradual_activation=True, sigma = None):
        if cb is None: cb = self.callback_cost_reductor1

        self.reductor_hist = []
        cb(self.reductor_params1)
        
        if method == 'cma':
            if sigma is None: sigma = self._epsilon

            from cma import fmin
            if gradual_activation:
                es = fmin(lambda x: min(self.cost_reductor1(x), self._confident_entries), self.reductor_params1, sigma, callback = cb, options=options)
            
            if not gradual_activation:
                es = fmin(self.cost_reductor1, self.reductor_params1, sigma, callback = cb, options=options)
            
            res = {}
            res['x'] = es[0]
            res['fun'] = es[1]
            res['evalsopt'] = es[2]
            res['evals'] = es[3]
            res['iterations'] = es[4]
            res['xmean'] = es[5]
            res['stds'] = es[6]

        else:
            res = minimize(lambda x: self.cost_reductor1(x), self.reductor_params1, method=method, callback=cb, options=options)
            c = self.cost_reductor1(res['x'])
        
        return res


    def optimize_reductor2(self, method = 'cma', options={'maxiter':150}, cb=None, sigma = None, gradual_activation = True):
        if cb is None:
            cb = self.callback_cost_reductor2

        self.reductor_hist2 = []
        cb(self.reductor_params2)
        if method.lower() == 'cma':
            from cma import fmin
            if sigma is None: sigma = self._epsilon
            if gradual_activation:
                es = fmin(lambda x: min(self.cost_reductor1(x), self._confident_entries), self.reductor_params1, sigma, callback = cb, options=options)
            
            if not gradual_activation:
                es = fmin(self.cost_reductor1, self.reductor_params1, sigma, callback = cb, options=options)
            
            res = {}
            res['x'] = es[0]
            res['fun'] = es[1]
            res['evalsopt'] = es[2]
            res['evals'] = es[3]
            res['iterations'] = es[4]
            res['xmean'] = es[5]
            res['stds'] = es[6]
        else:
            res = minimize(lambda x: self.cost_reductor2(x), self.reductor_params2, method=method, callback=cb, options=options)
            c = self.cost_reductor2(res['x'])
        
        return res


    def callback_cost_reductor1(self, reductor_params1):
        state, C, e = self.execute1(reductor_params1)

        cost = C +np.log(e)

        try: 
            self.reductor_hist1.append([C, cost])
        except:
            self.reductor_hist1 = []
            self.reductor_hist1.append([C, cost])


    def callback_cost_reductor2(self, reductor_params2):
        state, C, e = self.execute2(self.t1, self.t2, self.reductor_params1, reductor_params2)

        cost = C[1] + np.log(e[1])

        try: 
            self.reductor_hist2.append([C[1], cost])
        except:
            self.reductor_hist2 = []
            self.reductor_hist2.append([C[1], cost])

    def callback_cost_reductor_initialize1(self, reductor_params1):
        if type(reductor_params1) != np.ndarray:
            reductor_params1 = reductor_params1.ask(number=1, sigma_fac=0)[0]
        self.reductor_params1 = reductor_params1
        H, C, e = self.execute1(reductor_params1)

        cost = C + np.log(e)
        try: 
            self.initialize_hist1.append([C, cost])
        except:
            self.initialize_hist1 = []
            self.initialize_hist1.append([C, cost])

    
    def callback_cost_reductor_initialize2(self, reductor_params2):
        if type(reductor_params2) != np.ndarray:
            reductor_params2 = reductor_params2.ask(number=1, sigma_fac=0)[0]
        self.reductor_params2 = reductor_params2
        H, C, e = self.execute2(reductor_params2)

        cost = C + np.log(e)
        try: 
            self.initialize_hist2.append([C, cost])
        except:
            self.initialize_hist2 = []
            self.initialize_hist2.append([C, cost])


    def initialize1(self, target_params, max_steps, method='cma', options={'maxiter':100}):
        self.initialize_hist1 = []
        self.params2 = target_params
        t = np.linspace(0, 1, max_steps)
        for i in range(max_steps):
            C_ = self.execute(self.params1, self.params2, self.params3, self.reductor_params1, self.reductor_params2)[1][0]
            
            if C_ >= self._max_entries:
                self.params1 = target_params * t[i - 1]
                res = self.optimize_reductor1(method=method, options = options, cb=self.callback_cost_reductor_initialize1)

                yield res

        self.params1 = target_params
        res = self.optimize_reductor1(method=method, options=options, cb=self.callback_cost_reductor_initialize1)
        self.reductor_params2 = self.reductor_params1.copy()


        yield res


    def initialize2(self, target_params, max_steps, method='l-bfgs-b', options={'maxiter':100}):
        self.initialize_hist2 = []
        t = np.linspace(0, 1, max_steps)
        for i in range(max_steps):
            
            self.params2 = target_params * t[i]
            C_ = self.execute(self.params1, self.params2, self.params3, self.reductor_params1, self.reductor_params2)[1][1]
            if C_ >= self._max_entries:
                self.params2 = target_params2 * t[i - 1]
                res = self.optimize_reductor2(method=method, options = options, cb=self.callback_cost_reductor_initialize2)

                yield res

        res = self.optimize_reductor2(method=method, options=options, cb=self.callback_cost_reductor_initialize2)


        yield res



class cut_tfim(cut_model2):
    def __init__(self, qubits, input_layers, cb_layers, epsilon=0.01) -> None:
        super().__init__(qubits, input_layers, cb_layers, epsilon) 

        self.circuit2 = Circuit(self._qubits)
        self.circuit2.add(self.circuit_input_inside(end=True))
        self.params2 = 2 * np.pi * np.random.rand(len(self.circuit2.get_parameters()))

        self.circuit1 = Circuit(self._qubits)
        self.circuit1.add(self.circuit_input(end=False))
        self.params1 = 2 * np.pi * np.random.rand(len(self.circuit1.get_parameters()))

        self.circuit3 = Circuit(self._qubits)
        self.circuit3.add(self.circuit_input(end=True))
        self.params3 = 2 * np.pi * np.random.rand(len(self.circuit3.get_parameters()))

        self.zero_state = np.zeros(2**self._qubits, dtype=np.complex128)
        self.zero_state[0] = 1

        self.update_params1(self.params1)
        self.update_t1(1)
        self.get_final_state1()
        self.update_t2(1)


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

    def circuit_input_inside(self, end=False):
        for l in range(self.input_layers - self.cb_layers):
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

    def update_params1(self, params1):
        self.params1 = params1
        self.circuit1.set_parameters(self.params1)
        self.initial_state1 = self.circuit1.execute() 

      
    def initialize_reductor1(self, target_params, max_steps, method='l-bfgs-b', options={'maxiter':100}):
        self.update_params1(target_params)
        self.initialize_hist1 = []
        i_ = 1
        grad = np.zeros_like(self.reductor_params1)
        success = True
        for i in range(1, max_steps):
            if not success: 
                break
            self.update_t1(i / max_steps)
            C_, e = self.execute1(self.reductor_params1 + grad / max_steps)[1:]
            print(i, C_)
            if C_ >= self._max_entries:
                print("step", i, ":", C_, 'i_', i_)
                self.update_t1((i - 1) / max_steps)
                
                if i - 1 == i_ + 1: 
                    success = False
                    print('Unsuccessful activation')
                    res = {'status':'optimization failed'}
                    yield res

                else: 
                    print("step", i, ":", C_, 'i_', i_)
                    self.update_t1((i - 1) / max_steps)
                    i_ = i - 1
                
                if success: 
                    sigma = 0.1
                    self.reductor_params1 -= grad / max_steps
                    C_, e = self.execute1(self.reductor_params1 + grad / max_steps)[1:]
                    prev = self.reductor_params1.copy()
                    for i in range(20):
                        res = self.optimize_reductor1(method=method, options = options, cb=self.callback_cost_reductor_initialize1, sigma=sigma)
                        
                        if res['iterations'] > 10: 
                            print('success 1')
                            grad = res['x'] - prev
                            break
                        else: 
                            print('Try again with smaller sigma')
                            sigma = 0.75 * sigma
                    yield res

        if success:
            self.params1 = target_params
            res = self.optimize_reductor1(method=method, options=options, cb=self.callback_cost_reductor_initialize1)
            yield res


    
    def initialize_reductor2(self, target_params, max_steps, method='l-bfgs-b', options={'maxiter':100}):
        self.get_final_state1()
        self.initialize_hist2 = []
        i_ = 1
        grad = np.zeros_like(self.reductor_params2)
        success = True
        for i in range(1, max_steps):
            self.update_t2(i / max_steps)
            if not success: 
                break
            C_, e = self.execute2(self.reductor_params2 + grad / max_steps)[1:]
            print(i, C_)
            if C_ >= self._max_entries:
                print("step", i, ":", C_, 'i_', i_)
                if i - 1 == i_ + 1: 
                    success = False
                    print('Unsuccessful activation')
                    res = {'status':'optimization failed'}
                    yield res

                else: 
                    i_ = i - 1
                    self.update_t2((i - 1)/max_steps)
                
                if success: 
                    sigma = 0.1
                    self.reductor_params2 -= grad / max_steps
                    C_, e = self.execute2(self.reductor_params2)[1:]
                    prev = self.reductor_params2.copy()
                    for i in range(20):
                        res = self.optimize_reductor2(method=method, options = options, cb=self.callback_cost_reductor_initialize2, sigma=sigma)
                        
                        if res['iterations'] > 10: 
                            grad = res['x'] - prev
                            break
                        else: 
                            print('Try again with smaller sigma')
                            sigma = 0.75 * sigma
                    yield res

        if success:
            self.params2 = target_params

            res = self.optimize_reductor2(method=method, options=options, cb=self.callback_cost_reductor_initialize2)
            yield res