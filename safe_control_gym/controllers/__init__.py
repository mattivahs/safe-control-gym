'''Register controllers.'''

from safe_control_gym.utils.registration import register

register(idx='lqr',
         entry_point='safe_control_gym.controllers.lqr.lqr:LQR',
         config_entry_point='safe_control_gym.controllers.lqr:lqr.yaml')

register(idx='ilqr',
         entry_point='safe_control_gym.controllers.lqr.ilqr:iLQR',
         config_entry_point='safe_control_gym.controllers.lqr:ilqr.yaml')

register(idx='mpc',
         entry_point='safe_control_gym.controllers.mpc.mpc:MPC',
         config_entry_point='safe_control_gym.controllers.mpc:mpc.yaml')

register(idx='linear_mpc',
         entry_point='safe_control_gym.controllers.mpc.linear_mpc:LinearMPC',
         config_entry_point='safe_control_gym.controllers.mpc:linear_mpc.yaml')

register(idx='gp_mpc',
         entry_point='safe_control_gym.controllers.mpc.gp_mpc:GPMPC',
         config_entry_point='safe_control_gym.controllers.mpc:gp_mpc.yaml')

register(idx='pid',
         entry_point='safe_control_gym.controllers.pid.pid:PID',
         config_entry_point='safe_control_gym.controllers.pid:pid.yaml')

register(idx='ppo',
         entry_point='safe_control_gym.controllers.ppo.ppo:PPO',
         config_entry_point='safe_control_gym.controllers.ppo:ppo.yaml')

register(idx='sac',
         entry_point='safe_control_gym.controllers.sac.sac:SAC',
         config_entry_point='safe_control_gym.controllers.sac:sac.yaml')

register(idx='ddpg',
         entry_point='safe_control_gym.controllers.ddpg.ddpg:DDPG',
         config_entry_point='safe_control_gym.controllers.ddpg:ddpg.yaml')

register(idx='safe_explorer_ppo',
         entry_point='safe_control_gym.controllers.safe_explorer.safe_ppo:SafeExplorerPPO',
         config_entry_point='safe_control_gym.controllers.safe_explorer:safe_ppo.yaml')

register(idx='rarl',
         entry_point='safe_control_gym.controllers.rarl.rarl:RARL',
         config_entry_point='safe_control_gym.controllers.rarl:rarl.yaml')

register(idx='rap',
         entry_point='safe_control_gym.controllers.rarl.rap:RAP',
         config_entry_point='safe_control_gym.controllers.rarl:rap.yaml')

register(idx='pid_rl',
         entry_point='safe_control_gym.controllers.pid_rl.pid_rl:PIDRL',
         config_entry_point='safe_control_gym.controllers.pid_rl:pid_rl.yaml')

register(idx='cem',
         entry_point='safe_control_gym.controllers.cem.cem:CEMRL',
         config_entry_point='safe_control_gym.controllers.cem:cem.yaml')

register(idx='neuralmpc',
         entry_point='safe_control_gym.controllers.neuralmpc.neuralmpc:NeuralMPC',
         config_entry_point='safe_control_gym.controllers.neuralmpc:neuralmpc.yaml')

register(idx='fits',
         entry_point='safe_control_gym.controllers.fits.fits:FITS',
         config_entry_point='safe_control_gym.controllers.fits:fits.yaml')
