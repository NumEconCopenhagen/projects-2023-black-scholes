Python 3.8.5 (tags/v3.8.5:580fbb0, Jul 20 2020, 15:43:08) [MSC v.1926 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 3+4
7
>>> 
================================ RESTART: Shell ================================
>>> class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        H = HM**(1-par.alpha)*HF**par.alpha

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        pass    

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass


>>> plt.clf() 
u=5;
E=1000;
w = np.linspace(0, 100000,100000) 
f1= ((w*u)/(np.sqrt(1+(((w*u)/E)**2)))) 
plt.plot(w, f1)
axes = plt.gca()
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_xlim([0, 10E4])            # x-axis bounds
axes.set_ylim([0, 10E4])              # y-axis bounds
plt.show()
SyntaxError: multiple statements found while compiling a single statement
>>> SyntaxError: multiple statements found while compiling a single statement
SyntaxError: invalid syntax
>>> plt.clf() 
u=5;
E=1000;
w = np.linspace(0.25, 0.50,0.75;0.5,1.0,1.5) 
f1= ((w*u)/(np.sqrt(1+(((w*u)/E)**2)))) 
plt.plot(w, f1)
axes = plt.gca()
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_xlim([0, 10E4])            # x-axis bounds
axes.set_ylim([0, 10E4])              # y-axis bounds
plt.show()
SyntaxError: multiple statements found while compiling a single statement
>>> #2
>>> plt.clf() 
u=5;
E=1000;
w = np.linspace(0.8,0.9,1.0,1.1,1.2) 
f1= ((w*u)/(np.sqrt(1+(((w*u)/E)**2)))) 
plt.plot(w, f1)
axes = plt.gca()
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_xlim([0, 10E4])            # x-axis bounds
axes.set_ylim([0, 10E4])              # y-axis bounds
plt.show()
SyntaxError: multiple statements found while compiling a single statement
>>> plt.clf() 
u=5;
E=1000;
w = np.linspace(0.8,0.9,1.0,1.1,1.2) 
f1= ((w*u)/(np.sqrt(1+(((w*u)/E)**2)))) 
plt.plot(w, f1)
axes = plt.gca()
axes.set_xscale('log')
axes.set_yscale('log')
axes.set_xlim([0, 10E4])            # x-axis bounds
axes.set_ylim([0, 10E4])              # y-axis bounds
plt.show()
SyntaxError: multiple statements found while compiling a single statement
>>> #5
>>> class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        H = HM**(1-par.alpha)*HF**par.alpha

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        pass    

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):


#we will impliment a portfolio optimization,like for example if they have investment in the stock market
	# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
SyntaxError: expected an indented block
>>> #{'GOOG': 0.03835,
 'AAPL': 0.0689,
 'FB': 0.20603,
 'BABA': 0.07315,
 'AMZN': 0.04033,
 'GE': 0.0,
 'AMD': 0.0,
 'WMT': 0.0,
 'BAC': 0.0,
 'GM': 0.0,
 'T': 0.0,
 'UAA': 0.0,
 'SHLD': 0.0,
 'XOM': 0.0,
 'RRC': 0.0,
 'BBY': 0.01324,
 'MA': 0.35349,
 'PFE': 0.1957,
 'JPM': 0.0,
 'SBUX': 0.01082}

Expected annual return: 30.5%
Annual volatility: 22.2%
Sharpe Ratio: 1.28
SyntaxError: unexpected indent
>>> from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


latest_prices = get_latest_prices(df)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))12 out of 20 tickers were removed
Discrete allocation: {'GOOG': 1, 'AAPL': 4, 'FB': 12, 'BABA': 4, 'BBY': 2,
                      'MA': 20, 'PFE': 54, 'SBUX': 1}
Funds remaining: $11.8912 out of 20 tickers were removed
#so,with alpha=5,it doesn't help in the model
SyntaxError: multiple statements found while compiling a single statement
>>> #3
>>> ax = plt.axes()
ax.plot(w_f, w_m)
ax.set_title('2018 FIFA World Cup Final')
ax.set_ylabel('log(H_f/H_m')
ax.set_xlabel('log(w_f/w_m')
# Convert seconds-since-epoch numbers into struct_time objects and then to
# strings (you can use time.localtime() instead of time.gmtime() to get the
# time in your local timezone)
fmt = ticker.FuncFormatter(lambda x, pos: time.strftime('%H:%M', time.gmtime(x)))
ax.xaxis.set_major_formatter(fmt)

plt.show()
