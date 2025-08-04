import gym
import torch
from tqdm import tqdm
import wandb
import time
import numpy as np
import os
from halsi.rl.utils.mpi_tools import num_procs, mpi_fork, proc_id
from halsi.rl.agents.ppo import PPO    
from halsi.rl.agents.ppo_t_pen import PPOWithDuration   
from halsi.utils.general_utils import AttrDict
from halsi.models.diff import dynamicsVAE
from models.value_net import TrajectoryValueNet
import halsi.rl.envs
import math
import yaml

device = torch.device("cpu")

alpha = 0.8 

def get_obs(obs, env_name):
    if env_name == "FetchPyramidStack-v0": 
        out = torch.FloatTensor(np.concatenate((obs["observation"][:-6], obs["desired_goal"][-3:]))).unsqueeze(dim=0).to(device)
    else:
        out = torch.FloatTensor(np.concatenate((obs["observation"], obs["desired_goal"]))).unsqueeze(dim=0).to(device)
    return out

def logistic_fn(step, k=0.001, C=18000):
        return 1 / (1 + math.exp(-k * (step - C)))


def train(agent, step_actor, env, skill_vae, save_path):
    env_name = env.spec.id
    obs, ep_ret, ep_len = env.reset(), 0, 0
    o = get_obs(obs, env_name)

    env_step_cnt = 0

    local_steps_per_epoch = int(agent.steps_per_epoch / num_procs())

    for epoch in tqdm(range(agent.epochs)):
        for t in range(local_steps_per_epoch):
            n, duration_logits, v, logp, _, _ = agent.ac.step(o)
            duration_dist = torch.distributions.Categorical(logits=duration_logits)
            duration_idx = duration_dist.sample()
            logp_duration = duration_dist.log_prob(duration_idx)
            skill_duration = duration_idx.item() + 1

            if proc_id() == 0:
                wandb.log({"skill_duration": skill_duration}, env_step_cnt)

            obs_seq = o.unsqueeze(1)
            dummy_actions = torch.zeros((1, 1, skill_vae.n_actions))
            encoder_input = torch.cat([obs_seq, dummy_actions], dim=-1)
            with torch.no_grad():
                encoder_output = skill_vae.run_inference(encoder_input)
                mu = encoder_output[:, 0, :]

            residual_scale = logistic_fn(env_step_cnt)

            z = mu + residual_scale * n

            obs_seq = o.unsqueeze(1).repeat(1, skill_duration, 1)
            z_seq = z.unsqueeze(1).repeat(1, skill_duration, 1)
            decode_input = torch.cat([obs_seq, z_seq], dim=-1)

            with torch.no_grad():
                action_seq = skill_vae.diffusion_decoder.sample(decode_input, [skill_duration])

            skill_r, skill_done = 0, False

            for i in range(skill_duration):
                decoder_a = action_seq[0, i].detach().cpu().numpy()
                curr_o = get_obs(obs, env_name)
                s_plus_decoded_a = torch.cat((curr_o.to(device), torch.tensor(decoder_a).unsqueeze(0).to(device)), dim=1)

                step_a, step_v, step_logp, _, _ = step_actor.ac.step(s_plus_decoded_a)

                a_blend = alpha * decoder_a + (1 - alpha) * step_a[0].detach().cpu().numpy()

                a = np.clip(a_blend, env.action_space.low, env.action_space.high)

                obs, r, d, _ = env.step(a)

                env_step_cnt += 1
                ep_ret += r
                ep_len += 1
                skill_r += r

                step_actor.buf.store(
                    obs=s_plus_decoded_a.squeeze(0).cpu().numpy(),
                    act=step_a.squeeze(0).cpu().numpy(),
                    rew=r,
                    val=step_v,
                    logp=step_logp,
                    )

                if i == skill_duration - 1 or d:
                    last_input = s_plus_decoded_a.detach()

                if proc_id() == 0:
                    wandb.log({"action_x": float(a[0]), 
                               "action_y": float(a[1]),
                               "n_0": float(n[0, 0]),
                               "skill_reward": skill_r,
                               }, env_step_cnt)

                if d or (i == skill_duration - 1):
                    break

            if proc_id() == 0:
                wandb.log({"logistic_fn": residual_scale}, env_step_cnt)

            o2 = get_obs(obs, env_name)

            agent.buf.store(o.cpu().numpy(), n.cpu().numpy(), skill_r, v.cpu().numpy(), logp.cpu().numpy(),
                            logp_duration.cpu().numpy(), duration_idx.cpu().numpy())
            o = o2

            timeout = ep_len >= agent.max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, _, _, _, _ = agent.ac.step(o)
                    _, step_v, _, _, _ = step_actor.ac.step(last_input)
                else:
                    v, step_v = 0, 0

                v_val = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                step_v_val = step_v.detach().cpu().numpy() if isinstance(step_v, torch.Tensor) else step_v

                agent.buf.finish_path(v_val)
                step_actor.buf.finish_path(step_v_val)

                if terminal and proc_id() == 0:
                    wandb.log({"Episode Return": ep_ret}, env_step_cnt)

                obs, ep_ret, ep_len = env.reset(), 0, 0
                o = get_obs(obs, env_name)

        if proc_id() == 0 and ((epoch % agent.save_freq == 0) or (epoch == agent.epochs - 1)):
            torch.save(agent.ac, save_path)
            print(f"Saved model at epoch {epoch}")

        losses = agent.update()
        step_losses = step_actor.update()

        if proc_id() == 0:
            wandb.log({"pi_loss": losses.LossPi,
                       "v_loss": losses.LossV,
                       "entropy": losses.Entropy,
                       "step_pi_loss": step_losses.LossPi,
                       "step_v_loss": step_losses.LossV,
                       "step_entropy": step_losses.Entropy,
                       }, env_step_cnt)

def main():
    import argparse
    import yaml
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="table_cleanup/config_d2.yaml")
    parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")
    args=parser.parse_args()

    config_path = "configs/rl/" + args.config_file
    with open(config_path, 'r') as file:
        conf = yaml.safe_load(file)
        conf = AttrDict(conf)
    for key in conf:
        conf[key] = AttrDict(conf[key])

    mpi_fork(conf.setup.cpu)

    if proc_id() == 0:
        wandb.init(project=conf.setup.exp_name)
        wandb.run.name = conf.setup.env + "_stepwiseppo_seed_" + str(conf.setup.seed) + '_' + time.asctime().replace(' ', '_')

    env = gym.make(conf.setup.env)

    save_dir = "./results/saved_rl_models/" + args.dataset_name + "/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + "ppo_agent.pth"

    torch.set_num_threads(torch.get_num_threads())
    seed = conf.setup.seed + proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    skill_vae_path = "./results/saved_dskill_models/fetch_block_40000/skill_vae.pth"
    skill_vae = torch.load(skill_vae_path, map_location=torch.device('cpu'))
    skill_vae.eval()

    if not hasattr(skill_vae, 'decoder'):
        setattr(skill_vae, 'decoder', lambda x: skill_vae.diffusion_decoder.sample(x, [1]).squeeze(1))

    n_features = skill_vae.n_z
    n_obs = skill_vae.n_obs

    skill_agent = PPOWithDuration(
                gamma=conf.skill_agent.gamma, 
                seed=seed, 
                steps_per_epoch=conf.skill_agent.steps_per_epoch, 
                epochs=conf.setup.epochs,
                clip_ratio=conf.skill_agent.clip_ratio, 
                pi_lr=conf.skill_agent.pi_lr,
                vf_lr=conf.skill_agent.vf_lr, 
                train_pi_iters=conf.skill_agent.train_pi_iters, 
                train_v_iters=conf.skill_agent.train_v_iters, 
                lam=conf.skill_agent.lam, 
                max_ep_len=conf.setup.max_ep_len,
                target_kl=conf.skill_agent.target_kl, 
                obs_dim=n_obs, 
                act_dim=n_features, 
                max_duration=conf.skill_agent.max_duration)

    step_actor = PPO(ac_kwargs=dict(hidden_sizes=[conf.residual_agent.hid]*conf.residual_agent.l),
                    gamma=conf.residual_agent.gamma, 
                    seed=conf.setup.seed, 
                    steps_per_epoch=(conf.skill_agent.steps_per_epoch * 50), 
                    epochs=conf.setup.epochs,
                    clip_ratio=conf.residual_agent.clip_ratio, 
                    pi_lr=conf.residual_agent.pi_lr,
                    vf_lr=conf.residual_agent.vf_lr, 
                    train_pi_iters=conf.residual_agent.train_pi_iters, 
                    train_v_iters=conf.residual_agent.train_v_iters,
                    lam=conf.residual_agent.lam, 
                    target_kl=conf.residual_agent.target_kl, 
                    obs_dim=n_obs + env.action_space.shape[0], 
                    act_dim=env.action_space.shape[0], 
                    act_limit=1)

    if proc_id() == 0:
        print("Training step-wise PPO agent on CPU...")

    train(agent=skill_agent,
          step_actor=step_actor, 
          env=env,
          skill_vae=skill_vae,
          save_path=save_path)

if __name__ == '__main__':
    main()