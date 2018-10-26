import numpy as np
from metaq.envs.lunar_lander.lunar_lander import LunarLander as LunarLander_


class LunarLanderMultitask(LunarLander_):
    def __init__(self, task={}):
        if not 'helipad_center' in task or not 'height_variation' in task:
            # The default task
            super(LunarLanderMultitask, self).__init__(
                helipad_center=5,
                height_variation=0.0
            )
        else:
            super(LunarLander, self).__init__(
                helipad_center=task['helipad_center'],
                height_variation=task['height_variation']
            )
            
            
    def sample_tasks(self, num_tasks):
        tasks = []
        for _ in range(num_tasks):
            tasks.append(
                # {'helipad_center': np.random.randint(2, 9),
                #  'height_variation': np.random.uniform(-1.0, 1.0)}
                {'helipad_center': np.random.randint(4, 7),
                 'height_variation': np.random.uniform(0.0, 0.0)}
            )
        return tasks
    
    def reset_task(self, task):
        return self.set_task(task['helipad_center'], task['height_variation'])
        
    @property
    def _task(self):
        helipad_center, height_variation = self.current_task()
        return dict(
            helipad_center=helipad_center,
            height_variation=height_variation
        )
        