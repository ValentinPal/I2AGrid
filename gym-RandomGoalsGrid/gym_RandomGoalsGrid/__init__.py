from gym.envs.registration import register

register(
        id = 'RandomGoalsGrid-v0', 
        entry_point = 'gym_RandomGoalsGrid.envs:RandomGoalsGrid',
        )

register(
        id = 'RandomGoalsGrid3C-v0',
        entry_point = 'gym_RandomGoalsGrid.envs:RandomGoalsGrid3C',
        )

register(
        id = 'RandomGoalsGrid3CFast-v0',
        entry_point = 'gym_RandomGoalsGrid.envs:RandomGoalsGrid3CFast',
        )

register(
        id = 'RandomGoalsGridLinear-v0',
        entry_point = 'gym_RandomGoalsGrid.envs:RandomGoalsGridLinear',
        )
