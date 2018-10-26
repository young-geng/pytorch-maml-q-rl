import numpy as np
from metaq.envs.cartpole.cartpole import CartPoleEnv as CartPoleEnv_


class CartPoleMultitask(CartPoleEnv_):
    def __init__(self, task={}):
        if not 'goal_x' in task:
            # The default task
            super(CartPoleMultitask, self).__init__(goal_x=0.0)
        else:
            super(LunarLander, self).__init__(task['goal_x'])
            
            
    # def sample_tasks(self, num_tasks):
    #     tasks = []
    #     for goal_x in super(CartPoleMultitask, self).sample_tasks(num_tasks):
    #         tasks.append({'goal_x': goal_x})
    #     return tasks
    
    def sample_tasks(self, num_tasks):
        tasks = []
        for goal_x in np.random.choice([-0.5, 0.5], num_tasks):
            tasks.append({'goal_x': goal_x})
        return tasks
    
    def reset_task(self, task):
        self.set_task(task['goal_x'])
        return self.reset()
        
        
    def reset(self):
        return super(CartPoleMultitask, self).reset().astype(np.float32)
        
        
    def step(self, action):
        state, reward, done, info = super(CartPoleMultitask, self).step(action)
        return state.astype(np.float32), reward, done, info
        
    @property
    def _task(self):
        return dict(goal_x=self.goal_x)
        