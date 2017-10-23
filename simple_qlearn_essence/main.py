#!/usr/bin/env ipython

'''
# Simple Q-Learn Philosophy (?) Realization

## Task

We have a player, who's always hungry.

His hungriness is quantified into 100 levels.
  Starting at 100, for every second passed, he will become a little bit
  hungrier, i.e. hunriness -1 level. And if his hungriness decreases to 0,
  he'll cry for mom.

However we have 10 times to feed him, each time will +10 to hungriness.
  And our goal is to develop a policy, which is obvious, to keep him from
  calling mom as long as possible.

## Goal Policy

feed him every 10 seconds.

'''



import matplotlib.pyplot as plt
from random import random, gauss
import math

MAX_STEP = 20000


MAX_HUNGRINESS = 100
DIGEST_SPEED = 1
FOOD_CALORIES = 10


FRIDGE_CAPACITY = 10    # 10 chances to feed baby


# Baby
class Baby(object):
    '''
    Properties:
        _hp     - hunriness
        _vel    - hunriness to decrease by per sec
        _clock  - stop watch, will stop when he calls for mom
    '''

    def __init__(self, hungriness=MAX_HUNGRINESS,
                 digest_speed=DIGEST_SPEED,
                 fridge_capacity=FRIDGE_CAPACITY):
        '''
        Baby initializer

        Args:
            hunriness       - initial hungriness, default to 100
            digest_speed    - hunriness to decrease by per sec,
                              default to 1 (hunriness per sec)
        '''
        self._hp = hungriness
        self._vel = digest_speed
        self._clock = 0
        self._fridge = fridge_capacity


    def callmom(self):
        ''' see if he's calling mom '''
        if self._hp <= 0:
            return True


    def tick(self, time_span=1):
        ''' +1s '''
        self._hp -= time_span * self._vel
        self._clock += time_span
        return self.callmom()


    def feed(self, calories=FOOD_CALORIES):
        ''' feed him! '''

        if self._fridge <= 0:  # empty fridge
            return False

        self._hp += calories
        self._fridge -= 1
        if self._hp > MAX_HUNGRINESS:
            self._hp = MAX_HUNGRINESS
        return True


    @property
    def clock(self):
        ''' read from stop watch '''
        return self._clock



# Neccessary math func
def sigmoid(x):
    return (1. / ( 1. + math.exp(-x) ))



# Model
class Model(object):

    def __init__(self, ):
        self.Q = [ gauss(-1., 0.2) for _ in range(MAX_HUNGRINESS) ]
        self.Q[0] = -1  # Q of 100 hungriness is of no use


    def forward(self, hp):
        # added noise term
        return sigmoid(self.Q[hp]) + gauss(0, 0.5) > 0.5


    def predict(self, hp):
        return sigmoid(self.Q[hp]) > 0.5


    def backward(self, stat):

        hungriness = stat["hungriness"]
        action = stat["action"]
        score_board = stat["score_board"]["score"]
        diff = score_board[-1] - score_board[-2]  # positive if current policy
                                                  # is better
        for t in stat["time"]:
            curr_hp = hungriness[t-1]
            pre_Q = self.Q[curr_hp]
            update_delta = (
                (sigmoid(diff / MAX_HUNGRINESS) - 0.5) * 1e+39
                * (action["current"][curr_hp] - action["last"][curr_hp])
                * math.exp(t - stat["time"][-1])
            )
            self.Q[curr_hp] += update_delta
            # if update_delta != 0: print(update_delta)
            # if pre_Q != self.Q[curr_hp]: print("changed")




# Unit Test
if __name__ == '__main__':

    plot_data = {

        "time": [],

        "hungriness": [],

        "action": {
            "last": [ 0 for _ in range(MAX_HUNGRINESS) ],
            "current": [ 0 for _ in range(MAX_HUNGRINESS) ]
        },

        "score_board": {
            "step": [],
            "score": []
        }
    }


    Policy = Model()

    origin_Q = Policy.Q[:]

    # Running

    for step in range(MAX_STEP):

        # init runner
        Bob = Baby()

        # init statistics
        plot_data["time"] = []
        plot_data["hungriness"] = []
        plot_data["action"]["last"] = plot_data["action"]["current"]
        plot_data["action"]["current"] = [ 0 for _ in range(MAX_HUNGRINESS) ]

        while True:

            # Time goes by
            Bob.tick()

            # Take note
            plot_data["time"].append(Bob.clock)
            plot_data["hungriness"].append(Bob._hp)

            # Check if call for mom
            if Bob.callmom(): break

            # Feed
            if step != MAX_STEP-1:
                predict_fn = Policy.forward
            else:
                predict_fn = Policy.predict

            if predict_fn(Bob._hp):
                plot_data["action"]["current"][Bob._hp-1] = 1 if Bob.feed() else 0
            else:
                plot_data["action"]["current"][Bob._hp-1] = 0

        # He called for mom, gg

        plot_data["score_board"]["step"].append(step)
        plot_data["score_board"]["score"].append(Bob.clock
                                                 + sum(plot_data["hungriness"]))

        # Update policy
        if step > 0:
            Policy.backward(plot_data)

        # Output log message
        if step % 100 == 0:
            print("%dth iteration, stop-watch %d" % (step, Bob.clock))

        del Bob # clear memory by hand

    # End of Running


    # Final policy visualization
    plt.plot(plot_data["time"], plot_data["hungriness"])
    plt.show()

    # NOTE: the following plot is of no reference value,
    #       for noise terms are added into decision making function
    # plt.plot(plot_data["score_board"]["step"][::50],
    #          plot_data["score_board"]["score"][::50])
    # plt.show()

    # Policy weight visualization
    plt.plot([ h for h in range(MAX_HUNGRINESS) ], Policy.Q)
            #  [ x - y for x, y in zip(origin_Q, Policy.Q) ])
    plt.show()


    if plot_data["score_board"]["score"][-1] == 200:  # 200 - best policy
        print("WINNER WINNER! CHICKEN DINNER!")
