"""
Prob and Stats Lab – Discrete Probability Distributions

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 where required.
"""

import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    
    np.random.seed(42)

    #Step 2

    P_A=4/52
    P_B=4/52
    P_B_given_A=3/51
    p_A_intersection_B=(P_A)*(P_B_given_A)



    #Step 3 
    P_AB=(P_A)*(P_B)

    if p_A_intersection_B == P_AB:
        print("Independentant")


    else : 
        print("Dependant")


    count_A=0
    count_B_given_A=0 
    trials = 200000

    deck = np.array([1]*4 + [0] * 48)

    for _ in range(trials) :
        cards = np.random.choice(deck ,size=2 ,replace=False)

        if cards[0]==1:
            count_A +=1
            if cards[1]==1:
                count_B_given_A +=1

     #Emperical Results 

    empirical_P_A = count_A /trials 
    
    empirical_P_B_given_A=count_B_given_A / count_A

    absolute_error = abs(P_B_given_A - empirical_P_B_given_A)


    return (
     P_A,
     P_B,
     P_B_given_A,
     P_AB,
     empirical_P_A,
     empirical_P_B_given_A,
     absolute_error
)         
    """
    STEP 1: Consider a standard 52-card deck.
            Assume 4 Aces.

    STEP 2: Compute analytically:
            - P(A)
            - P(B)
            - P(B | A)
            - P(A ∩ B)

    STEP 3: Check independence:
            P(A ∩ B) ?= P(A)P(B)

    STEP 4: Simulate 200,000 experiments
            WITHOUT replacement.
            Use random_state=42.

            Estimate:
            - empirical P(A)
            - empirical P(B | A)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(B | A)
            empirical P(B | A)

    RETURN:
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    """

    raise NotImplementedError


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    
    np.random.seed(42)
    x=1
    p_X_1 = p**1*(1-p)**(1-1)

    p_X_0=p**0*(1-p)*(1-0)

    trials = 100000

    samples = np.random.binomial(n=1 , size=trials ,p=p)

    empeirical_p_X_1=np.mean(samples)

    error = abs(p_X_1 - empeirical_p_X_1)
    
    
    """
    STEP 1: Define Bernoulli(p) PMF:
            p_X(x) = p^x (1-p)^(1-x)

    STEP 2: Compute theoretical:
            - P(X = 1)
            - P(X = 0)

    STEP 3: Simulate 100,000 bulbs
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X = 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X = 1)
            empirical P(X = 1)

    RETURN:
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    """

    return p_X_1, p_X_0, empeirical_p_X_1, error

    raise NotImplementedError


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    
    np.random.seed(42)

    theoritical_P_0=math.comb(n,0)*p**0*(1-p)**(n-0)
    theoritical_P_2=math.comb(n,2)*p**2*(1-p)**(n-2)
    theoritical_P_ge_1=1 - math.comb(n,0)*p**0*(1-p)**(n-0)

    trials=100000

    samples=np.random.binomial(p=p , n=n ,size=trials)

    empirical_P_ge_1 = np.mean(samples >= 1)

    absolute_error=abs(theoritical_P_ge_1 - empirical_P_ge_1)
    """
    STEP 1: Define Binomial(n,p) PMF:
            P(X=k) = C(n,k)p^k(1-p)^(n-k)

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 2)
            - P(X ≥ 1)

    STEP 3: Simulate 100,000 inspections
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 1)
            empirical P(X ≥ 1)

    RETURN:
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    """

    return theoritical_P_0 , theoritical_P_2 , theoritical_P_ge_1 , empirical_P_ge_1 ,absolute_error

    raise NotImplementedError


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    
    np.random.seed(42)
    
    p = 1/6

    theoritical_P_1 = (5/6)**(1-1)*(1/6)
    theoritical_P_2 = (5/6)**(2-1)*(1/6)
    theoritical_P_3= (5/6)**(3-1)*(1/6)
    theoritical_P_4= (5/6)**(4-1)*(1/6)
    theoritical_P_gt_4 = 1 - ((theoritical_P_1) + (theoritical_P_2) +(theoritical_P_3) + (theoritical_P_4))

    trials =200000


    samples=np.random.geometric(p=1/6 , size=trials)

    empirical_P_gt_4 = np.mean(samples>4)

    absolute_error = abs(theoritical_P_gt_4 - empirical_P_gt_4)

    
    """
    STEP 1: Let p = 1/6.

    STEP 2: Define Geometric PMF:
            P(X=k) = (5/6)^(k-1)*(1/6)

    STEP 3: Compute theoretical:
            - P(X = 1)
            - P(X = 3)
            - P(X > 4)

    STEP 4: Simulate 200,000 experiments
            using random_state=42.

    STEP 5: Compute empirical:
            - empirical P(X > 4)

    STEP 6: Compute absolute error BETWEEN:
            theoretical P(X > 4)
            empirical P(X > 4)

    RETURN:
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    """

    return theoritical_P_1 , theoritical_P_3 ,theoritical_P_gt_4 ,empirical_P_gt_4 ,absolute_error

    raise NotImplementedError


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    
    np.random.seed(42)
    
    theoritical_P_0 = math.e**(-lam)* lam**0 / math.factorial(0)
    theoritical_P_15 = math.e**(-lam)* lam**15 / math.factorial(15)     

    array = np.array([])

    for i in range(0,18):
        array=np.append(array ,math.e**(-lam)* lam**i / math.factorial(i))

    theoritical_P_ge_18 = 1 - array.sum()
    trials = 100000

    samples=np.random.poisson(lam=lam , size =trials)

    emperical_P_ge_18=np.mean(samples >= 18)

    absolute_error = abs(theoritical_P_ge_18 - emperical_P_ge_18)



    """
    STEP 1: Define Poisson PMF:
            P(X=k) = e^(-λ) λ^k / k!

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 15)
            - P(X ≥ 18)

    STEP 3: Simulate 100,000 hours
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 18)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 18)
            empirical P(X ≥ 18)

    RETURN:
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    """

    return theoritical_P_0 , theoritical_P_15 , theoritical_P_ge_18 , emperical_P_ge_18 , absolute_error

    raise NotImplementedError
