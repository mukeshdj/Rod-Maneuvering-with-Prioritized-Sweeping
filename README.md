# Rod-Maneuvering-with-Prioritized-Sweeping

## Aim

To implement the **Prioritized Sweeping Reinforcement Learning algorithm** for a Rod Maneuvering environment and learn an optimal policy that efficiently guides the rod from its initial position to the target position while avoiding obstacles and minimizing the number of steps.

---

## Algorithm

### Prioritized Sweeping Algorithm

1. Initialize the state-value or action-value function (Q(s,a)) arbitrarily.

2. Initialize a model to store state transitions and rewards.

3. Observe the current state (s).

4. Select an action (a) using an exploration strategy (e.g., ε-greedy).

5. Execute the action and observe:

   * Next state (s')
   * Reward (r)

6. Store the transition ((s,a,r,s')) in the model.

7. Compute the priority:

   [
   P = |r + \gamma \max_a Q(s',a) - Q(s,a)|
   ]

8. If the priority exceeds a threshold, insert ((s,a)) into the priority queue.

9. While the queue is not empty:

   * Remove the state-action pair with the highest priority.
   * Update its Q-value.
   * Find predecessor states.
   * Recalculate their priorities.
   * Insert significant predecessors into the queue.

10. Repeat until the goal state is reached or the maximum number of episodes is completed.

11. Extract the optimal policy from the learned Q-values.

---

## Program

```python

#Rod Maneuvering with Prioritized Sweeping

import heapq
import numpy as np

# Grid size (Rod maneuvering environment)
ROWS, COLS = 5, 5

actions = [(-1,0),(1,0),(0,-1),(0,1)]  # Up, Down, Left, Right

gamma = 0.95
alpha = 0.1
theta = 0.01

Q = np.zeros((ROWS, COLS, len(actions)))

model = {}
predecessors = {}

priority_queue = []

goal = (4,4)

def step(state, action):
    r, c = state
    dr, dc = actions[action]

    nr = max(0, min(ROWS-1, r+dr))
    nc = max(0, min(COLS-1, c+dc))

    next_state = (nr, nc)

    reward = 100 if next_state == goal else -1

    return next_state, reward

for episode in range(100):

    state = (0,0)

    while state != goal:

        action = np.random.randint(len(actions))

        next_state, reward = step(state, action)

        model[(state, action)] = (next_state, reward)

        if next_state not in predecessors:
            predecessors[next_state] = set()

        predecessors[next_state].add((state, action))

        p = abs(
            reward +
            gamma*np.max(Q[next_state]) -
            Q[state][action]
        )

        if p > theta:
            heapq.heappush(priority_queue,
                           (-p, state, action))

        for _ in range(5):

            if not priority_queue:
                break

            _, s, a = heapq.heappop(priority_queue)

            ns, r = model[(s,a)]

            target = r + gamma*np.max(Q[ns])

            Q[s][a] += alpha*(target-Q[s][a])

            if s in predecessors:

                for ps, pa in predecessors[s]:

                    p = abs(
                        model[(ps,pa)][1] +
                        gamma*np.max(Q[s]) -
                        Q[ps][pa]
                    )

                    if p > theta:
                        heapq.heappush(
                            priority_queue,
                            (-p, ps, pa)
                        )

        state = next_state

print("Learning Complete")
print(Q)

```


## Output



<img width="824" height="666" alt="image" src="https://github.com/user-attachments/assets/f00f1c1e-ee53-497f-a878-8665196200ac" />



## Result

The **Prioritized Sweeping algorithm** was successfully implemented for the Rod Maneuvering environment. The agent learned an optimal path from the start state to the goal state by combining real experience with model-based planning. Prioritized updates enabled faster convergence and improved learning efficiency compared to conventional reinforcement learning approaches.


