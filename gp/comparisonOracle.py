class ComparisonOracle:
    def __init__(self, utility_function):
        self.utility_function = utility_function
        
    def compare(self, choice1, choice2):
        if self.utility_function(choice1) > self.utility_function(choice2):
            return choice1, choice2
        else:
            return choice2, choice1  

def sqare_root_utility(subset):
    return sum([x ** 0.5 for x in subset])

def cube_root_utility(subset):
    return sum([x ** (1./3) for x in subset])

def log_utility(subset):
    return sum([np.log(x) for x in subset])

def train(gp, af, max_q=50):
    for i in range(1, max_q+1): 
        print('-------------Train', i, '--------------')
        next1, next2 = af.get_next_point(dataset, gp)
        print('Optimal next pair is:', next1, next2)
        sup, inf = oracle.compare(next1, next2)
        dataset.add_single_comparison(sup, inf)
        print('User chooses:', sup)
        print('Updating GP with new data...')
        gp.update(dataset)
    print('---------Finished training!----------\n') 
    
def test(gp, af, test_size=50):
    gp_oracle = ComparisonOracle(gp.predict)
    success = 0
    for i in range(1, test_size+1):
        print('---------------Test', i, '---------------')
        test1 = np.random.randint(1, 50, 2)
        test2 = np.random.randint(1, 50, 2)
        gp_op = gp_oracle.compare(test1, test2)[0]
        oracle_op = oracle.compare(test1, test2)[0]
        print('Testing pair:', test1, test2)
        print('GP chooses', gp_op)
        print('User chooses', oracle_op)
        if np.array_equal(gp_op, oracle_op):
            success += 1
    print('-----------Finished testing!----------\n') 
    print("Accuracy: ", success/test_size)    




#### Simulation for pairwise-comparison experiment 
## Setup: 
##      - want to hire <= 100 people
##      - each subset is represented by [W, M] -> [number of women, number of men]
##      - capacity constraint: W + M <= 100
##      - value function defined on the subsets 
##      - true value function: square-root / cubic-root / log utility
## Goal:
##      - learn value function using GP + active learning (choosing optimal pair sequentially)
##      - Use as few query as possible (using 20 queries in the current example)

#### set seed for experiment

#### Create oracle w/ sqaure root utility, cubic or log utility
oracle = ComparisonOracle(cube_root_utility)

#### Create dataset object
dataset = Dataset(2)

#### Create query domain
input_domain = np.array([[x, 100 - x] for x in range(1, 100)])

#### GP + AF
gp = GP_pairwise(theta=20, seed=1) # build a new GP process
af = AcquisitionFunction(input_domain, seed=10) # build a new acquisition function

#### Training with active learning on 20 queries 
train(gp, af, 20)

#### Out-of-sample testing on 30 queries 
test(gp, af, 30)

